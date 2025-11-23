


# Calorie Burn Prediction Using Machine Learning Linear Regression Models


https://github.com/user-attachments/assets/97797f32-b3a6-4a39-ae48-871b7f1599c0


## **I. Giới thiệu project:**
### 1. Mục tiêu:
Mục tiêu của project là nghiên cứu và xây dựng một hệ thống thông minh có khả năng dự đoán lượng calories đốt cháy dựa trên dữ liệu hoạt động của người dùng, đồng thời tạo ra một nền tảng hỗ trợ đưa ra quyết định trong việc quản lý sức khỏe và luyện tập. Project tập trung triển khai đầy đủ quy trình Machine Learning từ xây dựng thuật toán mô hình, phân tích dữ liệu, xử lý đặc trưng đến huấn luyện và đánh giá mô hình, sau đó chuyển đổi mô hình đã xây dựng thành một ứng dụng web thực tế giúp người dùng tương tác trực tiếp với kết quả dự đoán. Thông qua hệ thống này, người dùng có thể dễ dàng lập kế hoạch vận động, điều chỉnh thói quen và hiểu rõ hơn về mức tiêu thụ năng lượng của bản thân. Mục tiêu cuối cùng là tạo ra một công cụ khoa học, dễ sử dụng và có giá trị thực tế, hỗ trợ quá trình theo dõi và tối ưu hoá lượng calories đốt cháy hằng ngày.

<p align="center">
<img src="https://media.tenor.com/cZplm9ukCgkAAAAj/hello-kitty-gym-workout.gif" alt="Demo GIF" width="40%">
</p>

### 2. Tính năng chính: 
- **Mô hình dự đoán lượng calories đốt cháy**: Mô hình Machine Learning được xây dựng để dự đoán lượng calories đốt cháy dựa trên dữ liệu hoạt động và thông tin cá nhân của người dùng. Mô hình được huấn luyện trên dữ liệu đã xử lý nhằm tạo ra dự đoán chính xác và ổn định, đóng vai trò làm nền tảng cho các tính năng của web app như Weekly Planner, Goal Translator hay What-if Coach.

- **Weekly planner**: Hệ thống tự động tạo lịch tập luyện theo tuần dựa trên hồ sơ người dùng và mục tiêu calories. Kế hoạch được phân bổ theo từng ngày với hoạt động, thời lượng và cường độ phù hợp, sao cho tổng calories đốt cháy sát nhất với mục tiêu và phù hợp với thời gian rảnh của người dùng.

- **Goal Translator**: Cho phép người dùng nhập mục tiêu tăng hoặc giảm cân và hệ thống tự động chuyển đổi thành lượng calories cần đốt mỗi tuần. Dựa trên đó, ứng dụng tạo kế hoạch tập luyện phù hợp về số buổi, thời lượng và cường độ để giúp người dùng đạt được mục tiêu.
- **Class Picker**: Mô hình được xây dựng dựa trên hồ sơ cá nhân của người dùng bao gồm cân nặng, nhóm hoạt động yêu thích, ngày trong tuần, và mục tiêu calories. Dựa trên các thông tin này, hệ thống sẽ tự động tính toán và gợi ý những hoạt động phù hợp nhất cho từng ngày, đồng thời ước tính lượng calories mà người dùng có thể đốt cháy khi thực hiện các hoạt động đó.
- **Meal suggest**: Tính năng Meal Suggest cung cấp danh sách món ăn phù hợp với nhu cầu calories và sở thích của người dùng. Hệ thống phân tích câu mô tả đầu vào, trích xuất thông tin quan trọng và tìm các món có mức độ tương đồng cao nhất trong dữ liệu món ăn để đưa ra gợi ý phù hợp.
- **What-if Coach**: What-if Coach giúp người dùng kiểm tra các kịch bản thay đổi thời lượng tập hoặc nhịp tim để xem lượng calories tiêu thụ sẽ thay đổi như thế nào. Dựa trên thông tin cá nhân và buổi tập hiện tại, hệ thống dự đoán calories và tạo bảng so sánh giữa các mức thời gian ±10 phút và nhịp tim ±20 bpm, giúp người dùng tối ưu hóa buổi tập một cách trực quan.
- **Swap Calories**: Tính năng này cho phép người dùng nhập tên món ăn và số lượng, sau đó hệ thống tự động tính toán cần tập bao nhiêu phút để đốt cháy lượng calories tương ứng. Ứng dụng lấy dữ liệu từ file món ăn, tính tổng calories, rồi dùng mô hình Machine Learning kết hợp tìm kiếm nhị phân để ước tính thời gian luyện tập chính xác và trả về kết quả gồm số phút cần thiết và mức calories đốt được.
### 3. Cách hoạt động tổng quát:
### 3.1. Xây dựng mô hình Linear Regression (from scratch):

- `BaseLinearRegression`  
  - Lớp cơ sở trừu tượng (abstract base class) cho tất cả các mô hình tuyến tính.  
  - Lưu trữ trọng số `W`, bias `b` và cờ `done_` để đánh dấu đã train xong.  
  - Định nghĩa:
    - `fit(X, y)`: hàm trừu tượng, mỗi mô hình con tự cài đặt.
    - `predict(X)`: áp dụng công thức \(ŷ = XW + b\) cho mọi mô hình con.

- `OlSLinearRegression`  
  - Cài đặt hồi quy tuyến tính **OLS** với nghiệm đóng (normal equation).  
  - Tham số `fit_intercept` cho phép chọn có/không dùng bias.  
  - Hàm `_design(X)` thêm cột 1 vào ma trận dữ liệu khi cần bias.  
  - Trong `fit`, dùng `np.linalg.lstsq` để giải hệ phương trình tuyến tính và tách bias/matrix trọng số.

- `GradientDescentLinearRegression`  
  - Cài đặt hồi quy tuyến tính bằng **gradient descent**:  
    - Khởi tạo `W` ngẫu nhiên nhỏ, `b = 0`.  
    - Định nghĩa hàm mất mát MSE và các bước `forward` / `backward` để tính dự đoán và gradient.  
    - Cập nhật `W`, `b` lặp lại nhiều vòng với `learning_rate`, dừng khi số vòng đạt `iterations` hoặc khi chênh lệch cost nhỏ hơn `convergence_tol`.

- `RidgeRegression`  
  - Cài đặt **Ridge** (L2-regularization) với nghiệm đóng:  
    - Thêm cột bias vào `X`.  
    - Giải nghiệm đóng với ma trận `w* = (X^T X + λ I)^(-1) X^T y`, trong đó không regularize bias.  
  - Phù hợp khi muốn giảm overfitting bằng phạt L2 lên trọng số.

- `LassoRegression`  
  - Cài đặt **Lasso** (L1-regularization) bằng **coordinate descent**:  
    - Khởi tạo toàn bộ trọng số bằng 0, bias bằng trung bình của `y`.  
    - Lần lượt cập nhật từng trọng số \( w_j \) theo quy tắc soft-thresholding với tham số `lam`.  
    - Dừng khi tổng thay đổi trọng số nhỏ hơn ngưỡng `tol` hoặc đạt `max_iter`.

- `ElasticNetRegression`  
  - Kết hợp cả L1 và L2 (Elastic Net), cũng bằng **coordinate descent**:  
    - Tham số `lam1` cho phần L1, `lam2` cho phần L2.  
    - Mỗi bước cập nhật một hệ số \( w_j \) với cả hai loại phạt.  
    - Dừng khi chuẩn sai khác giữa trọng số mới và cũ nhỏ hơn `tol` hoặc hết số vòng lặp.
### 3.2. Dữ liệu, tiền xử lý và train trong notebook:
- Notebook `ml.ipynb` là nơi thực hiện toàn bộ quy trình machine learning:
  - Tải / đọc dữ liệu thô (dùng `download_data.py` nếu cần download dữ liệu) và lưu vào thư mục `data/`.
  - Làm sạch dữ liệu: xử lý giá trị thiếu, loại bỏ ngoại lệ, chuẩn hoá đơn vị.
  - Xây dựng thêm các đặc trưng (BMI, cường độ, các feature phụ trợ cho mô hình…).
  - Gọi các mô hình tự xây dựng trong `Linear_Regression.py` để huấn luyện với data.
  - Đánh giá mô hình bằng các chỉ số (RMSE, R²).
  - Đóng gói và lưu mô hình vào thư mục `artifacts/`.
### 3.3. Các module tính năng (logic ứng dụng):
- `class_picker.py`: Dùng rule-based keyword matching để phân loại từng hoạt động vào nhóm (trong nhà/ngoài trời/thể thao/nghệ thuật/kháng lực), sau đó sort theo “Calories per kg” rồi random có kiểm soát (chọn 1 bài nặng, 1 bài nhẹ, rồi bổ sung) để lấy danh sách hoạt động đại diện cho mỗi nhóm.
- `generate_kitty_tip.py`: Dùng model calories + thông tin buổi tập (thời lượng, nhịp tim, target kcal) để tính kcal đốt, so sánh với mục tiêu và sinh ra một câu “lời nhắn Kitty” động viên/gợi ý điều chỉnh (tăng phút, tăng nhịp tim…).
- `goal_translator.py`: Nhận mục tiêu cân nặng (muốn tăng/giảm bao nhiêu kg/tuần), đổi sang mục tiêu kcal/tuần (dựa trên 1kg ≈ 7700 kcal), rồi gọi bộ `weekly_planner` để sinh lịch tập phù hợp.
- `meal_suggest.py`: Dùng TF-IDF + cosine similarity để so khớp câu hỏi người dùng với mô tả món ăn, kết hợp thêm tiêu chí calories, và trả về top món ăn phù hợp nhất.
- `swap_calo.py`: Cho phép đổi qua lại giữa calories của đồ ăn và thời lượng/cường độ tập: nhập số kcal muốn đốt (hoặc món ăn), hệ thống tính ra cần tập bao lâu / với nhịp tim nào.
- `weekly_planner.py`: Từ target kcal/tuần + profile người dùng, nó phân bổ mục tiêu calories theo từng ngày (pattern đều, hình kim tự tháp…) rồi tính gợi ý thời lượng và nhịp tim cho mỗi buổi tập trong tuần.
- `what_if.py`: Cho phép người dùng “giả lập” thay đổi (ví dụ tăng thêm buổi, kéo dài thời gian, đổi hoạt động) và tính lại tổng calories đốt/tuần để xem nếu thay đổi thì mục tiêu có đạt được không.
### 3.4. Xây dựng web: 

## **II. Đóng góp của từng thành viên:**
| Thành viên              | Mã số sinh viên | Công việc                                                                                                                                     | Tỷ lệ    |
|-------------------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|----------|
| **Nguyễn Khánh Duy**    | 11245870         | Thực hiện train model trong phần machine learning.<br> Xây dựng features cho web: goal translator, weekly planner, meal suggest              | **16.7%** |
| **Trần Thị Nhật Khánh** | 11234568         | Làm web                                                                                                                                       | **16.7%** |
| **Vũ Đức Anh**          | 11234569         | Tìm kiếm dữ liệu và thực hiện feature engineering trong machine learning.<br> Xây dựng features cho web: class picker, meal suggest.<br> Viết README | **16.7%** |
| **Lê Phạm Khánh Linh**  | 11234570         | Làm web                                                                                                                                       | **16.7%** |
| **Hoàng Thục Nhi**      | 11234570         | Làm web                                                                                                                                       | **16.7%** |
| **Vũ Trần Cát Linh**    | 11234570         | Phân tích và báo cáo dữ liệu (EDA) trong phần machine learning.<br> Xây dựng features cho web: what if coach, swap calorie                   | **16.7%** |



## **III. Hướng dẫn cài đặt và chạy code:**

### 1. Phiên bản Python sử dụng
Project được phát triển và chạy trên phiên bản:

```bash
Python 3.13.7
```
### 2. Cài đặt các thư viện cần thiết

Cài đặt toàn bộ thư viện bằng:

```bash
pip install -r requirements.txt
```
### 3. Cài đặt các file dữ liệu cần thiết:
Cài đặt toàn bộ data bằng cách chạy chương trình:

```bash
download_data.py
```
### 4. Cách chạy chương trình: Phần này hướng dẫn cách chạy web Django của project
### a. Cấu trúc Django
```md
kitty_fitness/
│
├── kitty_fitness/ # cấu hình chính
│ ├── settings.py
│ ├── urls.py
│
├── tracker/ # app chính
│ ├── models.py # (optional – project này dùng ML thay vì DB models)
│ ├── views.py # xử lý request + trả về HTML
│ ├── api.py # xử lý toàn bộ API gọi ML model
│ ├── urls.py
│ ├── templates/tracker/ # giao diện HTML
│ └── static/tracker/ # CSS, JS, hình
│
└── manage.py
```
### Phân tách giao diện và xử lý
- **views.py** – render HTML (Goal Translator, Weekly Planner…)
- **api.py** – xử lý logic tính toán, gọi ML model và trả JSON
- **urls.py** – định tuyến
- **static/** – chứa CSS, JS, hình ảnh
- **templates/** – HTML + Django Template Engine

### b. Các giao diện chính trong dự án
Django backend cung cấp nhiều API endpoint phục vụ các tính năng ML.

### Tất cả API đều sử dụng `POST` và trả về `JSON`.

| Tính năng       | Endpoint                | Ý nghĩa                                             |
| --------------- | ----------------------- | --------------------------------------------------- |
| Goal Translator | `/goal_translator_api/` | Tính toán kcal mục tiêu + lịch tập luyện            |
| Weekly Planner  | `/weekly_planner_api/`  | Sinh lịch tập theo tuần dựa trên profile người dùng |
| Meal Suggest    | `/meal_suggest_api/`    | Gợi ý món ăn từ dữ liệu similarity                  |
| What-if Coach   | `/what_if_api/`         | So sánh calories khi thay đổi thời gian/nhịp tim    |
| Swap Calories   | `/swap_calories_api/`   | Tính số phút cần tập để đốt calories từ một món ăn  |
| Class Picker    | `/class_picker_api/`    | Gợi ý loại bài tập phù hợp (nếu bạn triển khai sau) |

### c. Frontend — Backend Workflow

**Ví dụ: Goal Translator**

1. Người dùng nhập mục tiêu tăng/giảm cân  
2. Nhấn “Tạo kế hoạch” → JavaScript gửi payload bằng Fetch API  
3. Backend nhận request → xử lý tại `goal_translator_api`  
4. API thực hiện:
   - tính kcal cần đốt mỗi ngày  
   - gọi ML model để estimate calories  
   - sinh lịch tập theo ngày  
5. API trả JSON  
6. Frontend nhận JSON → dựng bảng kế hoạch  

> Meal Suggest, What-if Coach, Swap Calories dùng workflow tương tự.
### d. Tích hợp mô hình ML vào Django

Tất cả ML model được load sẵn ở backend trong `api.py`.
```python
import joblib
calories_model = joblib.load("models/calories_predictor.pkl")
```
Quy trình chung của mỗi API ML:

Nhận dữ liệu từ frontend

Xử lý dữ liệu (chuẩn hoá, scale…)

Dự đoán bằng ML model

Trả về JSON để hiển thị

### e. Giao diện (Frontend)

**Thư mục:**
```bash
tracker/templates/tracker/*.html
tracker/static/tracker/css/
tracker/static/tracker/js/
```
**HTML gồm:**

- form nhập thông tin  
- bảng kết quả  
- popup giới thiệu nhóm  
- table động bằng JavaScript  

**JavaScript gồm:** fetch API, tạo bảng kế hoạch, popup, update UI  
**CSS:** theme Hello Kitty, responsive, icon minh hoạ
### f. Tính năng chọn ngày rảnh (Free-days)

Payload từ frontend:
```json
{
  "free_days": ["Mon", "Wed", "Fri"]
}
```
### g. Ghi chú chân trang

Tất cả trang có footer chuẩn gồm:

- Mục tiêu (kcal)  
- Gợi ý thời lượng  
- Gợi ý nhịp tim  
- Kcal ước tính  
- Khả thi  
- Notes  

Có thể tách thành template component tái sử dụng.
### h. Session / User Profile (mở rộng)

Hiện dùng dữ liệu mẫu:
```makefile
age: 25
sex: male
height: 170
weight: 65
```
Mở rộng trong tương lai:

- User model  
- Lưu profile & lịch tập  
- Dashboard tiến độ  
### i. Chạy dự án
Frontend tự gọi API ML → hiển thị kết quả ngay.



