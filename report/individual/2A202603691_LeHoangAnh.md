# Báo cáo cá nhân — Day 01

## Lọc tin nhắn rác bằng ML, Deep Learning, Embedding và LLM

- **MSSV:** 2A202603691
- **Họ và tên:** Le Hoang Anh
- **SEED:** 7543
- **Thử thách cá nhân:** Đề #2 — thay đổi kích thước vocabulary BoW
- **Nguồn số liệu:** notebook đã lưu và `lab1_report.json` trong thư mục bài nộp.

## 1. Tóm tắt mức độ hoàn thành

Các phần tự cài đặt chính của lab đã có trong notebook. Báo cáo cuối notebook ghi **15/15 kiểm tra đạt**, **10/10 dự đoán có dấu thời gian**, và đã lưu kết quả benchmark cùng thử thách vocab. Bộ test repository chạy **8 passed, 1 skipped**; test bị skip là phần side lab tùy chọn chưa có `solution/solution.py`, không thuộc điểm chính theo rubric.

Bài nộp đã có đủ bốn tệp bắt buộc: báo cáo này, notebook có output, JSON kết quả và ảnh tính tay. Tuy nhiên, notebook còn một số ô trả lời chưa điền (đặc biệt giải thích ML, câu hỏi tổng hợp Pareto và bonus `MY_CASES`); do đó không nên xem bài là hoàn tất tuyệt đối theo rubric dù các hàm và kiểm tra kỹ thuật đã đạt. Điểm phần này cần được bổ sung bằng câu trả lời của người học trước khi nộp chính thức.

## 2. Thiết lập và cách đọc số liệu

Notebook sử dụng SMS Spam Collection. Bảng so sánh notebook ghi khoảng 5.572 tin ban đầu, tỷ lệ spam xấp xỉ 13%; tập test thường có khoảng 775 tin. Các kết quả LLM được đo trên **SUB cân bằng gồm 40 tin**, không phải toàn bộ test thường. Vì vậy, chỉ cột **F1 (SUB 40)** mới so sánh LLM với các mô hình khác trên cùng bộ tin; mỗi lỗi trên SUB có thể làm điểm F1 đổi đáng kể.

Các số latency/throughput là phép đo của một phiên Colab cụ thể, không phải cam kết hiệu năng triển khai. `USD / 1k tin` của các dòng LLM trong JSON bằng 0; đây không phải bằng chứng API miễn phí. Provider trong notebook là `deepseek`, nên trường chi phí không phản ánh hóa đơn thực tế hoặc bảng giá OpenAI. Cần lấy usage và giá của đúng provider để kết luận chi phí tiền tệ.

## 3. Kết quả trên test thường và SUB 40

| Phương án | Accuracy test | Precision | Recall | F1 test | F1 SUB 40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression tự code | 0.9690 | 0.9882 | 0.7850 | 0.8750 | 0.8235 |
| BoW + MLP | 0.9781 | 0.9688 | 0.8692 | 0.9163 | 0.9189 |
| MiniLM + kNN | 0.9768 | 0.9009 | 0.9346 | 0.9174 | 1.0000 |
| MiniLM + Logistic Regression | 0.9871 | 0.9533 | 0.9533 | **0.9533** | 1.0000 |
| MiniLM + MLP head | 0.9845 | 0.9524 | 0.9346 | 0.9434 | 0.9744 |
| LLM nhỏ, zero-shot | Chưa đo trên test thường | 1.0000* | 1.0000* | 1.0000* | 1.0000 |
| LLM lớn, zero-shot | Chưa đo trên test thường | 1.0000* | 1.0000* | 1.0000* | 1.0000 |
| LLM nhỏ, few-shot | Chưa đo trên test thường | 1.0000* | 0.9500* | 0.9744* | 0.9744 |

`*` Các chỉ số LLM trong kết quả chỉ đại diện cho SUB 40. Không được hiểu chúng là kết quả trên toàn bộ test thường. Trong 5 mô hình cục bộ, MiniLM + Logistic Regression có F1 test cao nhất (0.9533). Trên SUB 40, các mô hình MiniLM và hai cấu hình LLM zero-shot đều đạt F1 1.0, nhưng cỡ mẫu nhỏ không đủ để kết luận chúng tương đương hoặc mô hình nào tốt hơn một cách chắc chắn.

MLP cải thiện F1 so với BoW + Logistic Regression từ 0.8750 lên 0.9163, chủ yếu nhờ recall tăng từ 0.7850 lên 0.8692. Mạng hai lớp khi bỏ ReLU đạt F1 0.916 trên test, khớp với giải thích rằng hai phép biến đổi tuyến tính liên tiếp vẫn tương đương một phép biến đổi tuyến tính.

## 4. Tốc độ, độ trễ và giới hạn chi phí

| Phương án | Train/encode (s) | Throughput (tin/s) | p50 (ms/tin) | p95 (ms/tin) | Chi phí ghi trong JSON |
| --- | ---: | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression | 7.734 | 11,640.393 | 0.059 | 0.226 | $0 / 1k* |
| BoW + MLP | 6.334 | 37,549.186 | 0.324 | 0.458 | $0 / 1k* |
| MiniLM + kNN | 2.327 | 100.662 | 20.156 | 42.844 | $0 / 1k* |
| MiniLM + Logistic Regression | 4.439 | 890.346 | 29.953 | 43.007 | $0 / 1k* |
| MiniLM + MLP head | 4.469 | 1,342.493 | 12.980 | 21.563 | $0 / 1k* |
| LLM nhỏ, zero-shot | 0 | 1.057 | 956.806 | 1,203.316 | $0 / 1k* |
| LLM lớn, zero-shot | 0 | 1.108 | 906.011 | 1,222.589 | $0 / 1k* |
| LLM nhỏ, few-shot | 0 | 1.112 | 879.931 | 1,249.206 | $0 / 1k* |

`*` Giá trị 0 là giá trị benchmark được lưu, nhưng không đủ căn cứ để khẳng định chi phí API thực là 0. Với ngưỡng p95 < 50 ms, cả năm phương án cục bộ đều đạt trong phiên đo này; LLM không đạt. MiniLM + Logistic Regression có chất lượng test cao nhất trong nhóm cục bộ, còn BoW rất nhanh và nhẹ. Nếu chọn phương án cho hệ thống thật, cần benchmark trên phần cứng và tải mục tiêu, kiểm tra lại chất lượng trên dữ liệu đại diện, rồi mới chốt router hoặc ngưỡng phân loại.

## 5. Stress test 50 tin khó

| Phương án | F1 trên SUB / thường¹ | F1 trên 50 case | Accuracy trên 50 case |
| --- | ---: | ---: | ---: |
| BoW + Logistic Regression | 0.824 | 0.378 | 0.54 |
| BoW + MLP | 0.919 | 0.410 | 0.54 |
| MiniLM + kNN | 1.000 | 0.767 | 0.72 |
| MiniLM + Logistic Regression | 1.000 | 0.612 | 0.62 |
| MiniLM + MLP head | 0.974 | 0.653 | 0.66 |
| LLM nhỏ, zero-shot | 1.000 | **0.982** | **0.98** |
| LLM nhỏ, prompt phòng thủ | 1.000 | 0.964 | 0.96 |

¹ Notebook dùng SUB 40 cho LLM và hiển thị F1 SUB cho tất cả ở bảng stress test. Không đồng nhất cột này với F1 test thường của các mô hình cục bộ.

BoW giảm mạnh nhất trên bộ case khó: F1 từ 0.824/0.919 trên SUB xuống 0.378/0.410. MiniLM + kNN đạt F1 0.767; LLM zero-shot đạt 0.982 nhưng vẫn sai 1/50 case. Một case “spam giọng tự nhiên” (#16) bị tất cả mô hình đo trong ô này phân loại sai. Kết quả cho thấy benchmark cùng phân phối có thể lạc quan khi gặp obfuscation, thao túng, tin giao dịch và cách diễn đạt mới. Bộ 50 tin được viết tay hữu ích để tìm lỗi biên, nhưng không đại diện cho phân phối thực tế để tuyên bố độ chính xác ngoài đời.

## 6. Thử thách cá nhân — kích thước vocabulary

Notebook xác định SEED 7543 nhận Đề #2. Vocabulary được lấy từ train và Logistic Regression được huấn luyện lại cho từng cỡ từ vựng.

| Kích thước vocab | F1 test | Precision | Recall |
| ---: | ---: | ---: | ---: |
| 100 token | 0.8043 | 0.9610 | 0.6916 |
| 5.000 token | **0.8691** | 0.9881 | 0.7757 |

Tăng từ 100 lên 5.000 token làm F1 tăng **0.0648** (6.48 điểm phần trăm), precision tăng 0.0271 và recall tăng 0.0841. Trong thí nghiệm này, vocab nhỏ loại bỏ tín hiệu hữu ích nên bỏ sót nhiều spam hơn. Kết quả chỉ so sánh hai giá trị trên cùng lần chia dữ liệu; không chứng minh 5.000 luôn tối ưu. Cần dùng validation/cross-validation để chọn kích thước phù hợp.

## 7. Các điểm phương pháp cần lưu ý

- **Trùng lặp giữa train và test:** một SMS trùng ở hai tập làm đánh giá lạc quan vì mô hình có thể đã gặp chính ví dụ đó. Cần khử trùng lặp trước khi chia hoặc bảo đảm nhóm bản sao nằm cùng một phía.
- **Đánh giá trên dữ liệu huấn luyện:** không đo khả năng tổng quát hóa. Chỉ fit vocabulary, chuẩn hóa và mô hình trên train; báo cáo metric trên test độc lập.
- **Tìm láng giềng embedding:** dùng cosine thay vì dot product thô để giảm ảnh hưởng độ lớn vector; sắp xếp giảm dần để lấy các láng giềng giống nhất.
- **Parse nhãn LLM:** tìm chuỗi con `"spam"` có thể đọc sai câu phủ định như “not spam”. Cần parse nhãn có cấu trúc/ranh giới từ và kiểm thử trường hợp phủ định.

## 8. Tính tay và AI log

JSON ghi nhận đủ 5 giá trị tính tay: baseline accuracy **0.87**, sigmoid(0) **0.50**, F1 ví dụ **0.73**, cosine **0.60**, chi phí lý thuyết theo giả định đề bài **$10.50/ngày**. Ảnh tính tay được nộp riêng; rubric chấm các bước trên ảnh, không chỉ các giá trị JSON.

AI được dùng để hỗ trợ tổ chức/diễn đạt báo cáo và khung thử thách vocab; các số liệu thực nghiệm được đối chiếu với output notebook và JSON. Người nộp cần bảo đảm mô tả này phản ánh đúng phần AI đã hỗ trợ trong quá trình làm bài, vì quy định yêu cầu AI log trung thực.

## 9. Kết luận

Trong benchmark cục bộ này, MiniLM + Logistic Regression đạt F1 test cao nhất (**0.9533**) và p95 **43.007 ms**; BoW + Logistic Regression có p95 **0.226 ms** nhưng F1 test thấp hơn (**0.8750**). Trên bộ stress test viết tay, mọi mô hình giảm chất lượng, đặc biệt hai mô hình BoW. LLM nhỏ cho điểm stress test cao nhất nhưng p95 khoảng **1.20 giây**, không phù hợp yêu cầu 50 ms trong lần đo này; hơn nữa chi phí API chưa được đo đúng qua JSON.

Vì vậy, kết quả hiện tại gợi ý dùng mô hình cục bộ làm lớp xử lý chính và chỉ cân nhắc LLM cho một luồng chọn lọc sau khi đo chi phí, chất lượng và latency trên môi trường triển khai. Đây là kết luận từ một lần chạy và bộ stress test nhỏ, không phải bảo đảm hiệu năng sản phẩm.
