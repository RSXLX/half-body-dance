/**
 * ResultView — Phase 2 second slice.
 *
 * Pure render functions returning HTML strings (mirrors SetupView's split).
 * Score formatting is exposed separately so the existing pose_viewer.html can
 * keep its own renderer while we migrate; both produce identical strings.
 */

export interface ResultViewState {
  status: 'pending' | 'completed-no-score' | 'completed';
  finalAverageText: string;
  finalConclusionText: string;
  leftArmText: string;
  rightArmText: string;
  beatHitRateText: string;
  beatCompositeText: string;
  beatStatsText: string;
  matchLabel: string;
}

export interface ResultViewCallbacks {
  onReplay: () => void;
  onChangeAction: () => void;
  onBackToPractice: () => void;
  onOpenBeatAnalysis?: () => void;
}

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function formatPercent(value: number | null | undefined, fractionDigits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '--';
  return `${value.toFixed(fractionDigits)}%`;
}

export function formatRate(hits: number, total: number): string {
  if (total <= 0) return '--';
  return `${Math.round((hits / total) * 100)}%`;
}

export interface SubtitleParts {
  finalAverageText: string;
  beatHitRateText: string;
  beatCompositeText: string;
}

export function buildSubtitle(parts: SubtitleParts): string {
  return `过程均分 ${parts.finalAverageText} · 拍点命中 ${parts.beatHitRateText} · 拍点击分 ${parts.beatCompositeText}`;
}

export function reviewBody(state: ResultViewState): string {
  if (state.status === 'completed') {
    return `左右臂结果 ${state.leftArmText} / ${state.rightArmText}，动作结论 ${state.matchLabel}，节奏统计 ${state.beatStatsText}。`;
  }
  if (state.status === 'completed-no-score') {
    return `拍点命中 ${state.beatHitRateText}，拍点击分 ${state.beatCompositeText}。如需动作评分，请返回准备页打开摄像头后再练一轮。`;
  }
  return '进入练习页并完成一轮播放后，这里会输出动作和节奏复盘摘要。';
}

export function reviewTitle(state: ResultViewState): string {
  if (state.status === 'completed') return state.finalConclusionText || '已完成结算';
  if (state.status === 'completed-no-score') return '未产生有效评分';
  return '还没有结算结果';
}

export function heroTitle(state: ResultViewState): string {
  if (state.status === 'completed') return '本轮评分已结算';
  if (state.status === 'completed-no-score') return '本轮播放已完成';
  return '等待练习完成';
}

export function heroSubtitle(state: ResultViewState): string {
  if (state.status === 'completed') {
    return buildSubtitle({
      finalAverageText: state.finalAverageText,
      beatHitRateText: state.beatHitRateText,
      beatCompositeText: state.beatCompositeText,
    });
  }
  if (state.status === 'completed-no-score') {
    return '标准动作已完整播放，但本轮没有生成有效评分。通常是因为未打开摄像头或人体未稳定入镜。';
  }
  return '完成一轮播放后，这里会展示过程均分、节奏命中和动作结论。';
}

export function renderResultHero(state: ResultViewState): string {
  return `
    <article class="result-hero app-card">
      <span class="eyebrow">本轮结果</span>
      <h2 id="resultTitle">${escapeHtml(heroTitle(state))}</h2>
      <p id="resultSubtitle">${escapeHtml(heroSubtitle(state))}</p>
      <div class="result-banner">
        <strong id="finalAverageScore">${escapeHtml(state.finalAverageText)}</strong>
        <strong id="finalConclusion">${escapeHtml(state.finalConclusionText)}</strong>
      </div>
    </article>
  `;
}

export function renderResultGrid(state: ResultViewState): string {
  return `
    <div class="result-grid">
      <article class="app-card result-score-card">
        <div class="info-list">
          <div class="info-row"><span>左臂最终分</span><strong id="leftArmScore">${escapeHtml(state.leftArmText)}</strong></div>
          <div class="info-row"><span>右臂最终分</span><strong id="rightArmScore">${escapeHtml(state.rightArmText)}</strong></div>
          <div class="info-row"><span>拍点命中率</span><strong id="beatHitRate">${escapeHtml(state.beatHitRateText)}</strong></div>
          <div class="info-row"><span>拍点击分</span><strong id="beatCompositeScore">${escapeHtml(state.beatCompositeText)}</strong></div>
        </div>
      </article>
      <article class="app-card result-review">
        <div class="stack">
          <span class="dock-kicker">本轮复盘</span>
          <strong id="resultReviewTitle">${escapeHtml(reviewTitle(state))}</strong>
          <p id="resultReviewText">${escapeHtml(reviewBody(state))}</p>
        </div>
      </article>
    </div>
  `;
}

export function renderResultFooter(state: ResultViewState): string {
  const replayDisabled = state.status === 'pending' ? 'disabled' : '';
  return `
    <article class="result-footer">
      <button id="resultReplay" class="primary" type="button" ${replayDisabled}>再来一遍</button>
      <button id="resultToSetup" class="secondary" type="button">换一个动作</button>
      <div class="result-inline-links">
        <button id="resultToPractice" class="text-link" type="button" ${replayDisabled}>回到练习页</button>
        <button id="resultToBeatAnalysis" class="text-link" type="button">查看节拍分析</button>
      </div>
    </article>
  `;
}

export function renderResultView(state: ResultViewState): string {
  return `
    <section class="result-view" data-view="result">
      ${renderResultHero(state)}
      ${renderResultGrid(state)}
      ${renderResultFooter(state)}
    </section>
  `;
}

export function bindResultEvents(root: HTMLElement, callbacks: ResultViewCallbacks): () => void {
  const onClick = (event: Event) => {
    const target = event.target as HTMLElement | null;
    if (!target) return;
    if (target.closest('#resultReplay')) callbacks.onReplay();
    else if (target.closest('#resultToSetup')) callbacks.onChangeAction();
    else if (target.closest('#resultToPractice')) callbacks.onBackToPractice();
    else if (target.closest('#resultToBeatAnalysis')) callbacks.onOpenBeatAnalysis?.();
  };
  root.addEventListener('click', onClick);
  return () => root.removeEventListener('click', onClick);
}
