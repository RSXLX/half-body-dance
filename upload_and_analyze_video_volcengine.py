import argparse
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from analyze_video_emoji_volcengine import (
    DEFAULT_BANK_PATH,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    build_analysis_prompt,
    build_payload,
    build_result,
    extract_json_from_text,
    extract_output_text,
    load_prompt_bank,
    run_curl,
)


def load_tos_sdk():
    try:
        import tos  # type: ignore

        return tos
    except ImportError as exc:
        raise RuntimeError(
            "缺少 TOS Python SDK。请先执行: python3 -m pip install tos"
        ) from exc


def build_object_key(local_video: Path, prefix: str | None) -> str:
    utc_now = datetime.now(timezone.utc)
    date_prefix = utc_now.strftime("%Y/%m/%d")
    safe_prefix = (prefix or "video-emoji-analysis").strip("/ ")
    unique_suffix = uuid.uuid4().hex[:12]
    return f"{safe_prefix}/{date_prefix}/{local_video.stem}-{unique_suffix}{local_video.suffix.lower()}"


def guess_content_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def create_tos_client(tos_module, access_key: str, secret_key: str, endpoint: str, region: str):
    return tos_module.TosClientV2(
        ak=access_key,
        sk=secret_key,
        endpoint=endpoint,
        region=region,
    )


def upload_local_video(
    *,
    local_video: Path,
    access_key: str,
    secret_key: str,
    bucket: str,
    endpoint: str,
    region: str,
    object_key: str,
):
    tos = load_tos_sdk()
    client = create_tos_client(tos, access_key, secret_key, endpoint, region)
    client.put_object_from_file(
        bucket,
        object_key,
        str(local_video),
        content_type=guess_content_type(local_video),
    )
    return client, tos


def build_presigned_get_url(client, tos_module, bucket: str, object_key: str, expires: int) -> str:
    output = client.pre_signed_url(
        tos_module.enum.HttpMethodType.Http_Method_Get,
        bucket,
        object_key,
        expires=expires,
    )
    return output.signed_url


def main() -> int:
    parser = argparse.ArgumentParser(
        description="本地 MP4 自动上传到火山 TOS，并调用 doubao-1-5-thinking-vision-pro 输出 emoji 动作时间线。"
    )
    parser.add_argument("local_video", help="本地 MP4 文件路径")
    parser.add_argument(
        "--ark-api-key",
        default=os.environ.get("ARK_API_KEY"),
        help="ARK API Key；默认读取环境变量 ARK_API_KEY",
    )
    parser.add_argument(
        "--tos-access-key",
        default=os.environ.get("TOS_ACCESS_KEY") or os.environ.get("VOLCENGINE_ACCESS_KEY"),
        help="TOS Access Key；默认读取 TOS_ACCESS_KEY 或 VOLCENGINE_ACCESS_KEY",
    )
    parser.add_argument(
        "--tos-secret-key",
        default=os.environ.get("TOS_SECRET_KEY") or os.environ.get("VOLCENGINE_SECRET_KEY"),
        help="TOS Secret Key；默认读取 TOS_SECRET_KEY 或 VOLCENGINE_SECRET_KEY",
    )
    parser.add_argument(
        "--tos-bucket",
        default=os.environ.get("TOS_BUCKET"),
        help="TOS 存储桶名；默认读取环境变量 TOS_BUCKET",
    )
    parser.add_argument(
        "--tos-region",
        default=os.environ.get("TOS_REGION") or "cn-beijing",
        help="TOS 区域，默认 cn-beijing",
    )
    parser.add_argument(
        "--tos-endpoint",
        default=os.environ.get("TOS_ENDPOINT"),
        help="TOS Endpoint，默认按 region 推导为 tos-<region>.volces.com",
    )
    parser.add_argument(
        "--object-prefix",
        default="video-emoji-analysis",
        help="上传对象前缀目录，默认 video-emoji-analysis",
    )
    parser.add_argument(
        "--presign-expire",
        type=int,
        default=7200,
        help="预签名 URL 过期秒数，默认 7200",
    )
    parser.add_argument(
        "--delete-after",
        action="store_true",
        help="分析完成后删除 TOS 上的对象",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"ARK 模型名，默认 {DEFAULT_MODEL}")
    parser.add_argument("--ark-endpoint", default=DEFAULT_ENDPOINT, help="ARK responses endpoint")
    parser.add_argument(
        "--sampling-interval",
        type=int,
        default=None,
        help="视频采样间隔（秒）；不传则不发送 video_analysis_config",
    )
    parser.add_argument(
        "--max-sampling-frames",
        type=int,
        default=None,
        help="最大采样帧数；不传则不发送 video_analysis_config",
    )
    parser.add_argument(
        "--prompt-bank",
        default=str(DEFAULT_BANK_PATH),
        help="emoji prompt 库路径",
    )
    parser.add_argument(
        "--extra-guidance",
        help="附加动作分析要求",
    )
    parser.add_argument(
        "--ark-insecure",
        action="store_true",
        help="给调用 ARK 的 curl 添加 --insecure，用于本机 TLS 问题排查",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成上传对象 key 和 ARK payload，不实际上传或调用",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="uploaded_video_emoji_analysis.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    local_video = Path(args.local_video).expanduser().resolve()
    if not local_video.exists():
        raise SystemExit(f"本地视频不存在: {local_video}")

    tos_endpoint = args.tos_endpoint or f"tos-{args.tos_region}.volces.com"
    object_key = build_object_key(local_video, args.object_prefix)
    prompt_bank = load_prompt_bank(Path(args.prompt_bank))
    prompt_text = build_analysis_prompt(prompt_bank, args.extra_guidance)

    if args.dry_run:
        presigned_placeholder = f"https://{args.tos_bucket or 'your-bucket'}.{tos_endpoint}/{object_key}?signature=..."
        payload = build_payload(
            video_url=presigned_placeholder,
            model=args.model,
            sampling_interval=args.sampling_interval,
            max_sampling_frames=args.max_sampling_frames,
            prompt_text=prompt_text,
        )
        result = {
            "local_video": str(local_video),
            "tos_bucket": args.tos_bucket,
            "tos_region": args.tos_region,
            "tos_endpoint": tos_endpoint,
            "object_key": object_key,
            "presigned_expires_sec": args.presign_expire,
            "prompt_text": prompt_text,
            "payload": payload,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            import json

            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"dry-run 已输出到 {args.output}")
        return 0

    missing = []
    if not args.ark_api_key:
        missing.append("ARK_API_KEY / --ark-api-key")
    if not args.tos_access_key:
        missing.append("TOS_ACCESS_KEY / --tos-access-key")
    if not args.tos_secret_key:
        missing.append("TOS_SECRET_KEY / --tos-secret-key")
    if not args.tos_bucket:
        missing.append("TOS_BUCKET / --tos-bucket")
    if missing:
        raise SystemExit("缺少必要参数: " + ", ".join(missing))

    client = None
    tos_module = None
    uploaded_url = None
    response_json = None
    output_text = None
    parsed_json = None
    deleted_object = False

    try:
        client, tos_module = upload_local_video(
            local_video=local_video,
            access_key=args.tos_access_key,
            secret_key=args.tos_secret_key,
            bucket=args.tos_bucket,
            endpoint=tos_endpoint,
            region=args.tos_region,
            object_key=object_key,
        )
        uploaded_url = build_presigned_get_url(
            client,
            tos_module,
            args.tos_bucket,
            object_key,
            args.presign_expire,
        )

        payload = build_payload(
            video_url=uploaded_url,
            model=args.model,
            sampling_interval=args.sampling_interval,
            max_sampling_frames=args.max_sampling_frames,
            prompt_text=prompt_text,
        )
        response_json = run_curl(payload, args.ark_api_key, args.ark_endpoint, args.ark_insecure)
        output_text = extract_output_text(response_json)
        parsed_json = extract_json_from_text(output_text)

        result = build_result(
            video_url=uploaded_url,
            model=args.model,
            prompt_bank_path=Path(args.prompt_bank),
            prompt_text=prompt_text,
            response_json=response_json,
            output_text=output_text,
            parsed_json=parsed_json,
        )
        result["upload"] = {
            "local_video": str(local_video),
            "bucket": args.tos_bucket,
            "region": args.tos_region,
            "endpoint": tos_endpoint,
            "object_key": object_key,
            "presigned_expires_sec": args.presign_expire,
            "video_url": uploaded_url,
            "deleted_after_analysis": False,
        }

        if args.delete_after and client is not None:
            client.delete_object(args.tos_bucket, object_key)
            deleted_object = True
            result["upload"]["deleted_after_analysis"] = True

        with open(args.output, "w", encoding="utf-8") as f:
            import json

            json.dump(result, f, ensure_ascii=False, indent=2)

        analysis = result.get("analysis", {})
        segments = analysis.get("segments", [])
        print(f"上传完成: tos://{args.tos_bucket}/{object_key}")
        print(f"分析完成: {args.output}")
        print(f"视频摘要: {analysis.get('video_summary', '')}")
        print(f"动作段数: {len(segments)}")
        if segments:
            timeline = " ".join(
                f"[{seg['start_sec']:.2f}-{seg['end_sec']:.2f}s]{seg['emoji']}"
                for seg in segments
            )
            print(f"时间线: {timeline[:500]}")
        return 0
    finally:
        if args.delete_after and client is not None and not deleted_object:
            try:
                client.delete_object(args.tos_bucket, object_key)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
