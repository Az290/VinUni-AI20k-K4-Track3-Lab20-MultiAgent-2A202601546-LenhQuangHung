# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. Case nào nên dùng multi-agent? Vì sao?
2. Case nào không nên dùng multi-agent? Vì sao?

### Trả lời

**1. Case nào nên dùng multi-agent?**

Khi task có thể tách thành các sub-task đòi hỏi *loại năng lực khác nhau* mà nhồi chung vào
một prompt sẽ làm loãng context — ví dụ đúng như lab này: tìm nguồn (breadth, recall) vs.
phản biện độ tin cậy (critical reading, phát hiện mâu thuẫn/nguồn synthetic) vs. viết văn bản
mạch lạc có citation (structure, tone cho đúng audience). Trong benchmark thực đo được (xem
`reports/benchmark_report.md`), multi-agent cho citation coverage ổn định gần 100% vì Analyst
có bước riêng để đối chiếu claim với nguồn trước khi Writer viết — baseline một lượt dễ bỏ sót
bước này khi phải làm mọi thứ cùng lúc. Multi-agent cũng đáng dùng khi cần **khả năng debug
theo từng bước** (trace rõ ai làm gì, sai ở agent nào) — quan trọng trong môi trường production
nơi một câu trả lời sai cần truy ngược được nguồn gốc lỗi, thay vì chỉ có một lệnh gọi LLM hộp đen.

**2. Case nào không nên dùng multi-agent?**

Khi câu hỏi đơn giản, không cần nhiều bước xác minh, hoặc khi latency/cost là ràng buộc cứng
(ví dụ chatbot trả lời thời gian thực). Multi-agent trong lab này tốn nhiều lệnh gọi LLM hơn
hẳn (Researcher + Analyst + Writer + routing overhead so với 1 lệnh gọi baseline), nên latency
và cost luôn cao hơn — số liệu cụ thể nằm trong bảng so sánh của benchmark report. Nếu câu hỏi
không có nhiều nguồn mâu thuẫn cần phản biện (ví dụ định nghĩa một thuật ngữ, tóm tắt một tài
liệu duy nhất), bước Analyst gần như không thêm giá trị mà vẫn cộng thêm một lượt LLM — lúc đó
single-agent baseline vừa rẻ vừa nhanh hơn mà chất lượng không thua kém đáng kể. Quy tắc chung
áp dụng đúng như `README.md` đã nêu: "Không thêm agent nếu không có lý do rõ ràng."
