"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown, with a comparison table and analysis section."""

    lines = [
        "# Báo cáo Benchmark",
        "",
        "Tự động sinh bởi `multi_agent_research_lab.evaluation.report`. Mỗi dòng là số liệu "
        "trung bình trên bộ câu hỏi benchmark đã cấu hình (xem `configs/lab_default.yaml`).",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Ghi chú |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.append("")
    lines.append(_render_comparison(metrics))
    return "\n".join(lines) + "\n"


def _render_comparison(metrics: list[BenchmarkMetrics]) -> str:
    by_name = {m.run_name: m for m in metrics}
    baseline = by_name.get("baseline")
    multi = by_name.get("multi-agent")
    if not baseline or not multi:
        return ""

    lines = ["## So sánh Baseline vs Multi-Agent"]
    latency_delta = multi.latency_seconds - baseline.latency_seconds
    latency_pct = (
        (latency_delta / baseline.latency_seconds * 100) if baseline.latency_seconds else 0
    )
    lines.append(
        f"- **Latency**: multi-agent chậm hơn {latency_delta:+.2f}s ({latency_pct:+.0f}%) "
        "so với baseline."
    )

    if baseline.estimated_cost_usd is not None and multi.estimated_cost_usd is not None:
        cost_delta = multi.estimated_cost_usd - baseline.estimated_cost_usd
        cost_pct = (
            (cost_delta / baseline.estimated_cost_usd * 100) if baseline.estimated_cost_usd else 0
        )
        lines.append(
            f"- **Cost**: multi-agent tốn hơn ${cost_delta:+.4f} ({cost_pct:+.0f}%) so với "
            "baseline (nhiều lệnh gọi LLM hơn: researcher + analyst + writer so với 1 lệnh "
            "gọi baseline)."
        )

    if baseline.quality_score is not None and multi.quality_score is not None:
        quality_delta = multi.quality_score - baseline.quality_score
        lines.append(
            f"- **Quality**: multi-agent cao hơn {quality_delta:+.1f} điểm so với baseline."
        )

    if baseline.citation_coverage is not None and multi.citation_coverage is not None:
        cov_delta = multi.citation_coverage - baseline.citation_coverage
        lines.append(
            f"- **Citation coverage**: multi-agent cao hơn {cov_delta:+.0%} so với baseline."
        )

    lines.append(
        f"- **Failure rate**: baseline {baseline.failure_rate:.0%}, "
        f"multi-agent {multi.failure_rate:.0%}."
    )

    lines.append("")
    lines.append("## Phương pháp đo (Methodology)")
    lines.append(
        "Cả hai lần chạy đều dùng chung `SearchClient` và `LLMClient` (cùng nguồn dữ liệu "
        "offline/online, cùng model), để phép so sánh chỉ phản ánh khác biệt về *kiến trúc* "
        "(một lệnh gọi tổng quát vs. Supervisor điều phối Researcher → Analyst → Writer), "
        "không phải khác biệt công cụ."
    )

    lines.append("")
    lines.append("## Failure mode quan sát được")
    lines.append(
        "- **Guardrail chống lặp vô hạn của Supervisor**: nếu Researcher/Analyst liên tục lỗi "
        "(ví dụ LLM timeout), Supervisor vẫn ép trả về `done` khi đạt `max_iterations`, nên "
        "workflow luôn kết thúc với câu trả lời tốt nhất có thể thay vì treo vô hạn."
    )
    lines.append(
        "- **Search không trả về gì**: `SearchClient` (offline corpus hoặc Tavily) có thể trả "
        'về 0 nguồn với câu hỏi lệch chủ đề. Researcher ghi nhận `"No sources were found"` '
        "thay vì crash, các agent phía sau vẫn xử lý được dù thiếu bằng chứng."
    )
    lines.append(
        "- **LLM lỗi sau khi retry**: `LLMClient.complete` retry 3 lần với exponential "
        "backoff cho lỗi tạm thời (timeout/rate limit), sau đó raise `AgentExecutionError`. "
        "Mỗi worker agent bắt lỗi này và fallback sang nội dung tạm (ví dụ Writer ghép "
        "analysis/research notes lại) thay vì làm sập cả run — đây là lý do `failure_rate` "
        "có thể khác 0 dù `final_answer` vẫn tồn tại."
    )
    lines.append(
        "- **Cách fix đã áp dụng**: thêm try/except quanh mọi lệnh gọi LLM ở từng agent kèm "
        "nội dung fallback, cộng thêm `CriticAgent` như một validator xác định (deterministic) "
        "rẻ tiền để gắn cờ thiếu citation hoặc câu trả lời quá ngắn mà không tốn thêm lệnh "
        "gọi LLM."
    )
    return "\n".join(lines)
