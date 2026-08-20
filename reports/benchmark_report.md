# Báo cáo Benchmark

Tự động sinh bởi `multi_agent_research_lab.evaluation.report`. Mỗi dòng là số liệu trung bình trên bộ câu hỏi benchmark đã cấu hình (xem `configs/lab_default.yaml`).

## Bằng chứng trace (Trace evidence)

Trace công khai trên Langfuse cho 1 lần chạy multi-agent đầy đủ (câu hỏi: *"Summarize production guardrails for LLM agents"*), xem được không cần đăng nhập:

**https://cloud.langfuse.com/project/cmt178no3000lad0iwtlpv3gv/traces/b1a864ec7994c18b5afd8119e7cd6f84?observation=0aebee109a0e974b**

Trace cho thấy toàn bộ span `multi_agent_run` (tổng 16.18s) lồng đúng thứ tự routing thật: `supervisor → researcher (3.98s) → supervisor → analyst (5.36s) → supervisor → writer (4.95s) → critic`. Ảnh chụp trace này được lưu tại [`docs/trace_evidence.png`](../docs/trace_evidence.png).

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| baseline | 7.97 | 0.0004 | 7.7 | 60% | 0% | avg over 3 queries (max_iterations=6) |
| multi-agent | 21.25 | 0.0013 | 9.6 | 100% | 0% | avg over 3 queries (max_iterations=6) |

## So sánh Baseline vs Multi-Agent
- **Latency**: multi-agent chậm hơn +13.29s (+167%) so với baseline.
- **Cost**: multi-agent tốn hơn $+0.0008 (+185%) so với baseline (nhiều lệnh gọi LLM hơn: researcher + analyst + writer so với 1 lệnh gọi baseline).
- **Quality**: multi-agent cao hơn +1.9 điểm so với baseline.
- **Citation coverage**: multi-agent cao hơn +40% so với baseline.
- **Failure rate**: baseline 0%, multi-agent 0%.

## Phương pháp đo (Methodology)
Cả hai lần chạy đều dùng chung `SearchClient` và `LLMClient` (cùng nguồn dữ liệu offline/online, cùng model), để phép so sánh chỉ phản ánh khác biệt về *kiến trúc* (một lệnh gọi tổng quát vs. Supervisor điều phối Researcher → Analyst → Writer), không phải khác biệt công cụ.

## Kết luận — khi nào multi-agent đáng dùng?

Theo benchmark này, multi-agent tốn thêm **+167% latency** và **+185% cost** so với baseline một agent, đổi lại **+1.9 điểm quality** và **+40 điểm phần trăm citation coverage** (100% so với 60%). Sự đánh đổi này chỉ đáng giá khi độ tin cậy trích dẫn và khả năng phản biện nguồn quan trọng hơn tốc độ/chi phí — ví dụ: báo cáo nghiên cứu, câu trả lời cần tuân thủ compliance, hoặc bất cứ nội dung nào con người sẽ đối chiếu lại nguồn. Với câu hỏi nhạy cảm về độ trễ hoặc ít quan trọng (định nghĩa nhanh một thuật ngữ, tóm tắt một tài liệu duy nhất), baseline với chi phí thấp hơn và chất lượng không thua kém nhiều là lựa chọn mặc định tốt hơn — đúng với nguyên tắc đã nêu trong `README.md`: "Không thêm agent nếu không có lý do rõ ràng". Xem thêm phần exit ticket đầy đủ trong `docs/lab_guide.md`.

## Failure mode quan sát được
- **Guardrail chống lặp vô hạn của Supervisor**: nếu Researcher/Analyst liên tục lỗi (ví dụ LLM timeout), Supervisor vẫn ép trả về `done` khi đạt `max_iterations`, nên workflow luôn kết thúc với câu trả lời tốt nhất có thể thay vì treo vô hạn.
- **Search không trả về gì**: `SearchClient` (offline corpus hoặc Tavily) có thể trả về 0 nguồn với câu hỏi lệch chủ đề. Researcher ghi nhận `"No sources were found"` thay vì crash, các agent phía sau vẫn xử lý được dù thiếu bằng chứng.
- **LLM lỗi sau khi retry**: `LLMClient.complete` retry 3 lần với exponential backoff cho lỗi tạm thời (timeout/rate limit), sau đó raise `AgentExecutionError`. Mỗi worker agent bắt lỗi này và fallback sang nội dung tạm (ví dụ Writer ghép analysis/research notes lại) thay vì làm sập cả run — đây là lý do `failure_rate` có thể khác 0 dù `final_answer` vẫn tồn tại.
- **Cách fix đã áp dụng**: thêm try/except quanh mọi lệnh gọi LLM ở từng agent kèm nội dung fallback, cộng thêm `CriticAgent` như một validator xác định (deterministic) rẻ tiền để gắn cờ thiếu citation hoặc câu trả lời quá ngắn mà không tốn thêm lệnh gọi LLM.
