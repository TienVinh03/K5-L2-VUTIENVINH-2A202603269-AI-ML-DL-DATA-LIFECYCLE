# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4
 
**Python / PyTorch / Ultralytics:** Python 3.11 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id` = 468 ; `class_name` = cab ; `rank` = 1 ; `score` = 0.510915 ; `taxonomy_name` = ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Record này nói về **cả bức ảnh** (image-level classification).
   - **Vì sao:** Nhiệm vụ phân loại ảnh (`image_classification`) gán nhãn cho toàn bộ bức ảnh dựa trên ngữ cảnh tổng thể mà không xác định tọa độ hay hộp bao (bounding box) của từng đối tượng cụ thể. Mặc dù ảnh `traffic` chứa rất nhiều đối tượng khác nhau (xe buýt, ô tô con, người đi bộ...), mô hình chỉ tính toán phân phối xác suất và đưa ra danh sách các lớp dự đoán cho toàn bộ khung hình.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp đi kèm với **taxonomy / checkpoint** cố định được nạp từ trước (cụ thể là taxonomy `ImageNet-1K` gồm 1.000 lớp đi kèm file trọng số `yolo11n-cls.pt`). Mô hình không tự nghĩ ra tên lớp mà chỉ tính xác suất trên không gian danh sách nhãn có sẵn này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - Cần ghi đầy đủ cả 3 trường vì:
     + `taxonomy_name` xác định không gian ngữ nghĩa và hệ nhãn tham chiếu (tránh nhầm lẫn vì cùng tên lớp có thể có định nghĩa hoặc phạm vi khác nhau giữa các taxonomy như ImageNet-1K, COCO-80, OpenImages...).
     + `class_id` là mã định danh số học duy nhất dùng cho lưu trữ, xử lý lập trình và huấn luyện, tránh lỗi sai sót do chính tả hoặc chuyển ngữ.
     + `class_name` giúp con người (annotator, reviewer) đọc và hiểu nhãn dễ dàng.
     + Kết hợp cả ba giúp đảm bảo tính truy xuất nguồn gốc (provenance) và tái lập thực nghiệm chính xác.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Guideline cần nêu rõ **quy tắc ưu tiên chọn nhãn duy nhất (single-label)**, bao gồm:
     + Tiêu chí về diện tích và vị trí: Ưu tiên chọn vật thể chiếm diện tích lớn nhất hoặc nằm ở vị trí trung tâm / tiền cảnh của bức ảnh.
     + Tiêu chí chủ đề chính: Chọn đối tượng là trọng tâm truyền tải của khung hình.
     + Bảng thứ tự ưu tiên lớp (Class Priority Order): Quy định rõ nếu xuất hiện đồng thời cả vật A và vật B thì ưu tiên gán lớp nào trước.
     + Quy tắc escalation: Nếu các chủ thể ngang bằng nhau không thể phân định, chuyển ca khó lên Lab Coach thay vì tự ý chọn.
6. Vì sao không được lấy `score` của model làm ground truth?
   - Vì `score` chỉ là giá trị độ tin cậy toán học (confidence score / prediction probability) do mô hình tự tính toán, phản ánh mức độ "tin tưởng" của thuật toán chứ không đại diện cho sự thật khách quan (ground truth). Mô hình hoàn toàn có thể dự đoán sai hoặc bị ảo giác dù điểm score cao (ví dụ: dự đoán `cab` với score 0.51 trên ảnh giao thông đông đúc nhiều xe buýt). Ground truth bắt buộc phải do con người thẩm định, gán nhãn và kiểm duyệt theo quy tắc (guideline).

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name` = person ; `score` = 0.912625 ; `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92] ; `bbox_width` = 113.58 ; `bbox_height` = 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - **Góc trên-trái:** Tọa độ $(x_{min} = 385.33, y_{min} = 69.24)$ pixel, nằm ở đỉnh đầu của người đầu bếp (gốc tọa độ $(0, 0)$ là góc trên cùng bên trái bức ảnh).
   - **Góc dưới-phải:** Tọa độ $(x_{max} = 498.92, y_{max} = 348.92)$ pixel, nằm ở gót chân tiếp xúc mặt sàn của người đầu bếp.
   - **Độ rộng và chiều cao:** Hộp có chiều rộng là $113.58$ pixel ($498.92 - 385.33$) và chiều cao là $279.68$ pixel ($348.92 - 69.24$).
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - **Ngưỡng 0.60:** Có **6** prediction (`person: 0.91`, `bowl: 0.72`, `bowl: 0.70`, `oven: 0.69`, `oven: 0.63`, `person: 0.61`). Ở ngưỡng này chỉ giữ lại các vật thể có độ tin cậy cao và rõ ràng nhất.
   - **Ngưỡng 0.35:** Có **11** prediction (xuất hiện thêm các vật nhỏ/khuất như `bowl: 0.50`, `bowl: 0.46`, `cup: 0.45`, `cup: 0.38`, `bowl: 0.38`).
   - *(Ngưỡng 0.20)*: Có số lượng prediction tăng vọt (khoảng 16–18 prediction) do mô hình giữ lại nhiều hộp có độ tin cậy thấp ở các góc tối và dụng cụ treo trên tường.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Khi hạ ngưỡng, mô hình nhạy hơn nhưng sẽ kéo theo rất nhiều cảnh báo giả (false positives / box rác), box vẽ trùng lặp và nhãn gán sai.
   - Người **Reviewer/QC** sẽ phải tốn nhiều thời gian và công sức hơn để:
     + Rà soát, kiểm tra từng box xem có đối tượng thật hay chỉ là vùng nhiễu/bóng tối.
     + Bấm xóa (reject/delete) các box rác không đúng thực tế.
     + Nắn chỉnh lại biên box nếu mô hình vẽ quá rộng hoặc cắt lẹm vào vật thể.
     + Sửa lại nhãn phân loại nếu mô hình nhận diện sai tên lớp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - **Quy tắc:** *"Hộp bao (bounding box) phải ôm sát các điểm mép ngoài cùng của vật thể (tight bounding box), không để khoảng trống thừa giữa biên box và vật thể vượt quá 2 pixel; hộp phải bao trọn toàn bộ các phần nhìn thấy được của vật thể và tuyệt đối không cắt lẹm vào thân đối tượng."*
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - **Vật thể quan sát:** Ở góc dưới bên trái mép ảnh, có một phần cánh tay/bàn tay người (`person: 0.61`, $x_{min}=0.12$) bị mép ảnh cắt cụt; và chiếc bát (`bowl: 0.72`) ở góc dưới cùng bên trái bị mép ảnh và cạnh bàn cắt mất một phần.
   - **Lựa chọn:** Xử lý bằng **quy tắc trong Guideline**.
   - **Vì sao:** Việc vật thể bị cắt mép ảnh (truncation) hoặc che khuất (occlusion) là hiện tượng mang tính quy luật phổ biến trong nhiếp ảnh thực tế. Guideline cần chuẩn hóa quy tắc chung: *"Nếu vật thể bị cắt mép hoặc che khuất, annotator vẫn vẽ box ôm sát toàn bộ phần nhìn thấy được (visible part). Nếu phần nhìn thấy < 20% hoặc không đủ đặc trưng để con người nhận biết chắc chắn loại vật thể thì bỏ qua không gán nhãn"*. Việc ghi rõ vào guideline giúp annotator tự tin xử lý đồng bộ trên toàn bộ tập dữ liệu lớn mà không làm quá tải kênh escalation. Ta chỉ dùng escalation khi gặp vật thể biến dạng dị biệt, mập mờ hoàn toàn không thể đoán được danh tính.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record:
   - `instance_id` = kitchen-001
   - `class_name` = person
   - `score` = 0.899318
   - Số điểm polygon (`polygon_point_count`) = 348
   - 5 điểm đầu của `polygon_xy` = `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - Bounding box chỉ cung cấp một khung hình chữ nhật bao quanh biểu thị vị trí tọa độ cực trị $(x_{min}, y_{min}, x_{max}, y_{max})$, do đó chứa nhiều pixel nền thừa (background) hoặc pixel của các vật thể xung quanh.
   - Ngược lại, **polygon** (đa giác) cung cấp thông tin hình học chi tiết ở cấp độ điểm ảnh (pixel-level mask):
     + Mô tả chính xác đường bao, ranh giới và hình dáng hình học thực tế của vật thể (bất kể vật thể có hình dạng phức tạp, uốn lượn hay bị nghiêng).
     + Cho biết diện tích bề mặt thực tế và chu vi thực sự của đối tượng.
     + Phân tách triệt để từng pixel thuộc về vật thể khỏi vùng nền và khỏi các đối tượng khác xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - **Mục đích:** `instance_id` dùng để định danh duy nhất cho từng cá thể độc lập trong cùng một bức ảnh tĩnh (ví dụ: `kitchen-001`, `kitchen-002`...).
   - **Ý nghĩa:**
     + **Không phải class ID:** `class_id` là mã số đại diện cho cả một chủng loại/lớp đối tượng (ví dụ: tất cả mọi người trong ảnh đều chung `class_id = 0`, tất cả cái bát đều chung `class_id = 45`). Trong khi đó, `instance_id` phân biệt các cá thể riêng biệt trong cùng một lớp (hai người khác nhau trong cùng ảnh sẽ có hai `instance_id` khác nhau).
     + **Không phải tracking ID:** `tracking_id` là định danh theo dõi một vật thể liên tục xuyên suốt qua chuỗi các khung hình video (video frames) theo thời gian. Ngược lại, `instance_id` ở đây chỉ có phạm vi hiệu lực cục bộ trong một ảnh tĩnh đơn lẻ của bài lab, không có giá trị liên kết hay bảo toàn danh tính qua thời gian/video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - **Quy tắc:** *"Đường biên đa giác (polygon boundary) phải bám khít theo đường viền thực tế của vật thể ở cấp độ pixel; các đỉnh đa giác phải ôm sát ranh giới phân tách giữa vật thể và nền (dung sai không quá 1–2 pixel), không được cắt lẹm vào thân đối tượng và tuyệt đối không bao trùm pixel của nền hoặc của vật thể lân cận."*
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - **Vị trí quan sát trên ảnh `kitchen`:**
     + Hai vật chạm nhau: Các chảo/nồi kim loại treo lơ lửng phía trên bếp nấu chạm sát mép vào nhau; hoặc đáy của chiếc bát tiếp xúc trực tiếp với mặt bàn gỗ.
     + Chỗ biên mờ / bị che: Cánh tay người ở mép dưới bên trái bị mép ảnh cắt đứt và bóng tối che khuất viền cổ tay; hoặc chân đế của lò nướng màu đen chìm vào vùng tối dưới sàn bếp.
   - **Lựa chọn:** Xử lý bằng **quy tắc trong Guideline** (Guideline rule).
   - **Vì sao:** Hiện tượng hai vật tiếp xúc, chạm nhau hoặc chồng lấn (occlusion) là rất phổ biến trong cảnh sinh hoạt thực tế. Guideline cần chuẩn hóa quy tắc rõ ràng: *"Tại vị trí hai vật chạm nhau hoặc đè lên nhau, annotator phải vẽ đường viền bám sát đối tượng ở phía trước (tiền cảnh); vật thể bị che khuất phía sau sẽ bị ngắt biên tại mép che, không vẽ mask đè lấn lên nhau (non-overlapping masks)"*. Điều này giúp annotator có quy chuẩn xử lý nhất quán trên diện rộng. Ta chỉ dùng escalation khi gặp khu vực biên quá tối/mờ đến mức mắt người không thể nhận biết được ranh giới phân chia giữa hai vật.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn đơn cấp ảnh (`class_id`, `class_name`), không có tọa độ. | Ảnh `traffic` có nhiều đối tượng (xe buýt, ô tô, người đi bộ) nhưng model đoán `cab` (score 0.51) và `minibus` (score 0.16), gây khó khăn khi chọn 1 nhãn duy nhất. | Đọc kỹ guideline ưu tiên, chọn 1 lớp đại diện theo quy tắc diện tích/vị trí trung tâm; escalation nếu ngang hàng. | Đối chiếu nhãn được gán với toàn bộ ảnh xem có tuân thủ đúng quy tắc ưu tiên trong guideline không. |
| Phát hiện vật thể | Tọa độ bounding box (`bbox_xyxy` = `[x_min, y_min, x_max, y_max]`) kèm `class_name` cho từng vật. | Trên ảnh `kitchen`, các vật dụng nhỏ bị che khuất hoặc cắt mép bàn/tủ khiến việc xác định ranh giới mép box gặp khó khăn. | Vẽ box ôm sát từng vật thể (tight bounding box), gán đúng lớp COCO-80 theo quy định che khuất trong guideline. | Kiểm tra độ ôm sát của box, phát hiện vật thể bị bỏ sót (false negative) hoặc box vẽ thừa/sai đối tượng (false positive). |
| Instance segmentation | Tập hợp đỉnh đa giác khép kín (`polygon_xy`) kèm `class_name` và `instance_id` cho từng cá thể. | Trên ảnh `kitchen`, nhiều vật dụng đặt sát nhau, viền bóng mờ hoặc màu sắc tương đồng khiến đường biên dễ bị lẫn vào nền. | Chấm các điểm đa giác bám sát viền pixel thực tế, phân tách rạch ròi từng instance độc lập, gán đúng `instance_id`. | Soi độ mịn của đường biên mask, đảm bảo không lấn sang vật thể khác hoặc cắt lẹm vào đối tượng. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Chỉ sử dụng các ảnh mẫu công khai COCO đã được cấp phép (CC BY 2.0); tuyệt đối không tải lên, lưu trữ hoặc công khai thông tin định danh cá nhân (PII), họ tên, MSSV hay dữ liệu nhạy cảm trong mã nguồn, báo cáo và đầu ra của bài lab.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Dừng ngay thao tác và báo cáo trực tiếp cho Lab Coach / Giảng viên hướng dẫn để được hướng dẫn xử lý (escalation).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
