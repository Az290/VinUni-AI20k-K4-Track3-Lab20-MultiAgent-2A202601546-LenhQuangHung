# Design: Multi-Agent Research System

## Problem

Người dùng gửi một câu hỏi nghiên cứu dài (ví dụ: "Research GraphRAG state-of-the-art and
write a 500-word summary"). Hệ thống cần: (1) tìm nguồn liên quan, (2) đánh giá/so sánh độ
tin cậy và mâu thuẫn giữa các nguồn, (3) viết câu trả lời cuối cùng có trích dẫn rõ ràng trỏ
về nguồn. Yêu cầu production: không được chạy vô hạn, phải có thể trace lại từng bước, và
phải benchmark được so với một baseline đơn giản hơn.

## Why multi-agent?

Một agent đơn (single-agent) phải làm cùng lúc 3 việc có bản chất khác nhau trong một
lần gọi LLM: tìm/chọn nguồn, phản biện độ tin cậy, và viết văn bản mạch lạc. Khi nhồi cả
ba vào một prompt, context bị loãng — model có xu hướng bỏ qua bước đánh giá nguồn để nhảy
thẳng sang viết, dẫn đến citation coverage thấp hơn và ít khi chỉ ra được nguồn nào yếu/mang
tính synthetic. Tách vai trò cho phép:

- **Researcher** tập trung 100% vào việc thu thập và tóm tắt nguồn — prompt ngắn, rõ mục tiêu.
- **Analyst** chỉ nhận research_notes + danh sách nguồn, nhiệm vụ duy nhất là phản biện.
- **Writer** chỉ tổng hợp, không phải tự tìm/đánh giá nguồn.

Đánh đổi: nhiều lệnh gọi LLM hơn → latency và cost cao hơn baseline (xem
`reports/benchmark_report.md`). Multi-agent chỉ đáng dùng khi câu hỏi đủ phức tạp để phần
thưởng chất lượng (citation coverage, khả năng phát hiện nguồn yếu) bù lại chi phí đó.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Điều phối: đọc state, quyết định route tiếp theo hoặc dừng | `ResearchState` hiện tại | `route_history` (+1 route), `iteration` +1 | Không tự fail — là logic thuần state-inspection, guardrail `max_iterations` luôn ép `done` |
| Researcher | Tìm nguồn (Tavily hoặc offline corpus), tóm tắt thành research_notes có [n] citation | `request.query` | `state.sources`, `state.research_notes` | Search rỗng → notes rỗng nhưng không crash; LLM lỗi sau 3 retry → fallback dùng raw sources làm notes |
| Analyst | Đọc research_notes + sources, so sánh quan điểm, gắn cờ nguồn yếu/synthetic | `state.research_notes`, `state.sources` | `state.analysis_notes` | LLM lỗi → fallback pass-through research_notes nguyên văn |
| Writer | Tổng hợp research_notes + analysis_notes thành final_answer có citation + mục Sources | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` | LLM lỗi → fallback ghép analysis/research notes làm câu trả lời tạm |
| Critic (bonus) | Validate final_answer: citation coverage, độ dài tối thiểu, citation trỏ sai index | `state.final_answer`, `state.sources` | `state.trace` entry + `state.errors` nếu có finding | Không gọi LLM — thuần regex/deterministic nên không thể fail vì lý do bên ngoài |

## Shared state

`ResearchState` (xem [core/state.py](../src/multi_agent_research_lab/core/state.py)):

- `request: ResearchQuery` — câu hỏi gốc + audience + max_sources; không đổi trong suốt run,
  mọi agent đọc lại để không "quên" câu hỏi ban đầu.
- `iteration`, `route_history` — đếm số bước Supervisor đã ra quyết định; là guardrail
  chính chống loop vô hạn, và là bằng chứng trace cho thứ tự routing.
- `sources: list[SourceDocument]` — nguồn thô Researcher tìm được; Analyst và Writer đều
  cần đọc lại danh sách này để trích dẫn đúng index.
- `research_notes` / `analysis_notes` / `final_answer` — ba "khoang" riêng biệt để mỗi agent
  ghi kết quả của mình mà không ghi đè lên agent khác — đây là điều kiện Supervisor dùng để
  quyết định route (thiếu field nào thì gọi agent tương ứng).
- `agent_results: list[AgentResult]` — lưu token usage/cost theo từng agent, dùng để tính
  `estimated_cost_usd` trong benchmark.
- `trace: list[dict]` — mọi span (route, researcher, analyst, writer, critic) đều
  `add_trace_event` vào đây; đây chính là "hộp đen" khi cần debug sai ở bước nào.
- `errors: list[str]` — mọi lỗi (search fail, LLM fail, citation thiếu) được append vào đây
  thay vì raise exception làm sập cả workflow; benchmark dùng field này để tính `failure_rate`.

## Routing policy

Xem [agents/supervisor.py](../src/multi_agent_research_lab/agents/supervisor.py). Cây quyết
định thuần state-inspection (không cần gọi LLM để routing — rẻ và deterministic):

```text
if iteration >= max_iterations:       -> done   (guardrail cứng)
elif final_answer đã có:              -> done
elif sources rỗng HOẶC research_notes rỗng: -> researcher
elif analysis_notes rỗng:             -> analyst
else:                                 -> writer
```

Writer luôn dẫn tới Critic rồi kết thúc (graph edge cứng `writer -> critic -> END`), nên
ngay cả khi Supervisor lý thuyết có thể route sai, cấu trúc graph vẫn đảm bảo không lặp lại
sau khi có final_answer.

## Guardrails

- **Max iterations**: `Settings.max_iterations` (mặc định 6, đọc từ `.env`/`MAX_ITERATIONS`).
  Supervisor kiểm tra trước tiên trong `decide_route`; đây là guardrail duy nhất thực sự chặn
  vòng lặp vô hạn Supervisor ↔ Researcher/Analyst.
- **Timeout**: `Settings.timeout_seconds` (mặc định 60s) truyền vào `OpenAI` client và
  `httpx` cho Tavily — mỗi lệnh gọi mạng có deadline riêng, không phải timeout toàn workflow.
- **Retry**: `LLMClient._call` dùng `tenacity` retry 3 lần với exponential backoff cho
  `APITimeoutError`, `RateLimitError`, `APIError`.
- **Fallback**: mỗi worker agent (Researcher/Analyst/Writer) bọc lệnh gọi LLM trong
  try/except; khi retry cạn, agent trả về nội dung tạm (raw sources, pass-through notes,
  ghép notes) thay vì để exception lan lên làm crash cả graph.
- **Validation**: `CriticAgent` kiểm tra citation coverage, citation trỏ sai index, và độ dài
  tối thiểu của `final_answer`; không tốn thêm lệnh gọi LLM.

## Benchmark plan

Bộ 3 câu hỏi cấu hình trong [configs/lab_default.yaml](../configs/lab_default.yaml) (đồng bộ
với `evaluation/benchmark.py::run_benchmark_suite`):

1. "Research GraphRAG state-of-the-art and write a 500-word summary"
2. "Compare single-agent and multi-agent workflows for customer support"
3. "Summarize production guardrails for LLM agents"

Metric đo (xem `evaluation/benchmark.py`):

| Metric | Cách đo |
|---|---|
| Latency | wall-clock giây, đo bằng `perf_counter()` quanh toàn bộ run |
| Cost | tổng `cost_usd` từ token usage của mọi `AgentResult` (ước tính theo giá gpt-4o-mini) |
| Quality | heuristic 0-10: độ dài câu trả lời (tối đa 5đ) + citation coverage (tối đa 4đ) + 1đ base − phạt theo số lỗi ghi nhận; bổ sung cho điểm peer-review thủ công theo rubric |
| Citation coverage | tỉ lệ số source index [n] hợp lệ thực sự được trích dẫn trong final_answer / tổng số nguồn |
| Failure rate | tỉ lệ run có `state.errors` không rỗng |

Kỳ vọng: multi-agent latency/cost cao hơn baseline rõ rệt (nhiều lệnh gọi LLM hơn); citation
coverage và khả năng phát hiện nguồn yếu của multi-agent nên tốt hơn baseline nhờ Analyst có
bước phản biện riêng. Kết quả thực đo nằm trong `reports/benchmark_report.md`.
