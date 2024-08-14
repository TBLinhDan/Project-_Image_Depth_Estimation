## **Project: Image Depth Estimation** *(Phân tích thông tin chiều sâu của ảnh)*  
Thực hiện bài toán **Depth Information Reconstruction**, mô phỏng độ sâu ***(disparity map)*** từ một cặp ảnh stereo cho trước. Với Disparity Map, ta có thể biết được thông tin chiều sâu (Depth Information) thực tế, để hướng tới mục tiêu phục hồi kiến trúc 3D từ một cặp ảnh 2D. 
Trong project này, ta sẽ làm quen với bài toán **Stereo Matching** thông qua việc triển khai một số thuật toán tính Disparity Map từ cặp ảnh stereo cho trước. Stereo Matching là một bài toán lớn trong lĩnh vực Thị giác máy tính (Computer Vision), được sử dụng để xác định độ sâu của các đối tượng trong một cảnh khi so sánh hai (hoặc nhiều) hình ảnh được chụp từ các góc nhìn hơi khác nhau. Bằng cách phân tích sự khác biệt giữa các điểm tương ứng trong các hình ảnh này, thuật toán Stereo Matching ước tính khoảng cách đến các điểm khác nhau, tạo ra một biểu diễn 3D của cảnh.

### **Part A_**Problem 1: Xây dựng hàm tính ***disparity map*** của hai ảnh stereo đầu vào (ảnh bên trái (L) và ảnh bên phải (R)) theo phương thức ***pixel-wise matching***.  

**1. Tải cặp ảnh stereo Tsukuba**
* Tải thủ công và upload lên google drive
```
from google.colab import drive
drive.mount('/content/drive')
!cp /path/to/dataset/on/your/drive
```
* Dụng lệnh gdown
```
# Tsukuba: https://drive.google.com/file/d/14gf8bcym_lTcvjZQmg8kwq3aXkENBxMQ/view?usp=sharing
!gdown --id 14gf8bcym_lTcvjZQmg8kwq3aXkENBxMQ
!unzip tsukuba.zip -d tsukuba
```

**2. Sử dụng thư viện CV2 để đọc ảnh đầu vào:**

```
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
```
```
left_img_path = 'tsukuba/left.png'
right_img_path = 'tsukuba/right.png'
disparity_range = 16

left = cv2.imread(left_img_path)
right = cv2.imread(right_img_path)

cv2_imshow(left)
cv2_imshow(right)
```  
**3. Định nghĩa hàm tính cost:**

```
def l1_distance(x, y):
    return abs(x - y)

def l2_distance(x, y):
    return (x - y) ** 2
```
**4. Định nghĩa hàm **pixel_wise_matching_l1() để thực hiện:**
* Đọc ảnh chụp bên trái (left) và ảnh chụp bên phải (right) dưới dạng ảnh grayscale (ảnh mức xám) đồng thời ép kiểu ảnh về np.float32.
* Khởi tạo hai biến height, width có giá trị bằng chiều cao, chiều rộng của ảnh trái.
* Khởi tạo một ma trận không - zero matrix (depth) với kích thước bằng height, width.
* Với mỗi pixel tại vị trí (h, w) (duyệt từ trái qua phải, trên xuống dưới) thực hiện các bước sau:  
(a) Tính cost (L1 hoặc L2) giữa các cặp pixel left[h, w] và right[h, w - d] (trong đó d ∈ [0, disparity_range]). Nếu (w - d) < 0 thì gán giá trị cost = max_cost (max_cost = 255 nếu dùng L1 hoặc 2552 nếu dùng L2).  
(b) Với danh sách cost tính được, chọn giá trị d (doptimal) mà ở đó cho giá trị cost là nhỏ nhất.  
(c) Gán depth[h, w] = doptimal × scale. Trong đó, scale = 255 disparity_range  
(giá trị scale mặc định gán = 16)

```
def pixel_wise_matching_l1(left_img, right_img, disparity_range, save_result=True):
    # Read left, right images then convert to grayscale
    left  = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255

    for y in range(height):
        for x in range(width):
            # Find j where cost has minimum value
            disparity = 0
            cost_min  = max_value

            for j in range(disparity_range):
                cost = max_value if (x - j) < 0 \
                                else l1_distance(int(left[y, x]), int(right[y, x - j]))

                if cost < cost_min:
                    cost_min  = cost
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'pixel_wise_l1.png', depth)
        cv2.imwrite(f'pixel_wise_l1_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)
```

**5. Định nghĩa hàm pixel_wise_matching_l2()**:

```
def pixel_wise_matching_l2(left_img, right_img, disparity_range, save_result=True):
    # Read left, right images then convert to grayscale
    left  = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255 ** 2

    for y in range(height):
        for x in range(width):
            # Find j where cost has minimum value
            disparity = 0
            cost_min  = max_value

            for j in range(disparity_range):
                cost = max_value if (x - j) < 0 else l2_distance(int(left[y, x]), int(right[y, x - j]))

                if cost < cost_min:
                    cost_min  = cost
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'pixel_wise_l2.png', depth)
        cv2.imwrite(f'pixel_wise_l2_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth
```

**6. Disparity map Result:**

* **L1 result**
```
depth, color = pixel_wise_matching_l1(
    left_img_path,
    right_img_path,
    disparity_range,
    save_result=True
)
cv2_imshow(depth)
cv2_imshow(color)
```

* **L2 result**
```
depth = pixel_wise_matching_l2(
    left_img_path,
    right_img_path,
    disparity_range,
    save_result=True
)
cv2_imshow(depth)
cv2_imshow(color)
```

### **Part B:** Xây dựng hàm tính ***disparity map*** của hai ảnh stereo đầu vào (ảnh bên trái (L) và ảnh bên phải (R)) theo phương thức ***Window-based matching***.  

**1. Tải cặp ảnh stereo Aloe**
* Tải thủ công và upload lên google drive
```
from google.colab import drive
drive.mount('/content/drive')
!cp /path/to/dataset/on/your/drive .
```
* Dụng lệnh gdown
```
# Aloe: https://drive.google.com/file/d/1wxmiUdqMciuTOs0ouKEISl8-iTVXdOWn/view?usp=drive_link
!gdown --id 1wxmiUdqMciuTOs0ouKEISl8-iTVXdOWn
!unzip Aloe_images.zip
```

**2. Sử dụng thư viện CV2 để đọc ảnh đầu vào:**

```
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
```

**3. Định nghĩa hàm tính cost:**

```
def l1_distance(x, y):
    return abs(x - y)

def l2_distance(x, y):
    return (x - y) ** 2
```
**4. Định nghĩa hàm window_based_matching_l1() để thực hiện:**
* Đọc ảnh chụp bên trái (left) và ảnh chụp bên phải (right) dưới dạng ảnh grayscale (ảnh mức xám) đồng thời ép kiểu ảnh về np.float32.
* Khởi tạo hai biến height, width có giá trị bằng chiều cao, chiều rộng của ảnh trái.
* Khởi tạo một ma trận không - zero matrix (depth) với kích thước bằng height, width.
* Tính nửa kích thước của window tính từ tâm đến cạnh của window (có kích thước k x k) theo công thức kernel_half = (k - 1)/2 (lấy nguyên).  
* Với mỗi pixel tại vị trí (h, w) (h ∈ [kernel_half, height - kernel_half], w ∈ [kernel_half, width - kernel_half]; duyệt từ trái qua phải, trên xuống dưới), thực hiện các bước sau:  
(a) Tính tổng các cost (l1 hoặc l2) giữa các cặp pixel left[h + v, w + u] và right [h + v, w + u - d] (trong đó d ∈ [0, disparity_range] và u, v ∈ [-kernel_half, kernel_half]) nằm trong vùng window với tâm là vị trí của pixel đang xét. Nếu tại vị trí cho (w + u - d) < 0 thì gán giá trị cost của cặp pixel đang xét = max_cost (max_cost = 255 nếu dùng L1 hoặc 2552 nếu dùng L2).
(b) Với danh sách cost tính được, chọn giá trị d (doptimal) mà ở đó cho giá trị cost tổng là nhỏ nhất.  
(c) Gán depth[h, w] = doptimal × scale. Trong đó, scale = 255 / disparity_range  
 (gán mặc định giá trị scale = 3).

```
def window_based_matching_l1(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    # Read left, right images then convert to grayscale
    left  = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 * 9

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            # Find j where cost has minimum value
            disparity = 0
            cost_min  = 65534

            for j in range(disparity_range):
                total = 0
                value = 0

                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = l1_distance(
                                int(left[y + v, x + u]), int(right[y + v, (x + u) - j]))
                        total += value

                if total < cost_min:
                    cost_min = total
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'window_based_l1.png', depth)
        cv2.imwrite(f'window_based_l1_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth
```

**5. Định nghĩa hàm window_based_matching_l2()**:

```
def window_based_matching_l2(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    # Read left, right images then convert to grayscale
    left  = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 ** 2

    for y in range(kernel_half, height-kernel_half):
        for x in range(kernel_half, width-kernel_half):

            # Find j where cost has minimum value
            disparity = 0
            cost_min  = 65534

            for j in range(disparity_range):
                total = 0
                value = 0

                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        value = max_value
                        if (x + u - j) >= 0:
                            value = l2_distance(int(left[y + v, x + u]),  int(right[y + v, (x + u) - j]))
                        total += value

                if total < cost_min:
                    cost_min = total
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'window_based_l2.png', depth)
        cv2.imwrite(f'window_based_l2_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth
```

###**B_I. Disparity map Result:**

```
left_img_path = 'Aloe/Aloe_left_1.png'
right_img_path = 'Aloe/Aloe_right_1.png'
disparity_range = 64
kernel_size = 3

left = cv2.imread(left_img_path)
right = cv2.imread(right_img_path)

cv2_imshow(left)
cv2_imshow(right)
```

* **L1 result**

```
depth = window_based_matching_l1(
    left_img_path,
    right_img_path,
    disparity_range,
    kernel_size=kernel_size,
    save_result=True
)

cv2_imshow(depth)
```

* **L2 result**

```
depth = window_based_matching_l2(
    left_img_path,
    right_img_path,
    disparity_range,
    kernel_size=kernel_size,
    save_result=True
)

cv2_imshow(depth)
```

### **B_II. "invariant to linear changes" Problem**

Thay đổi cặp ảnh đầu vào là ***Aloe_left_1.png*** và ***Aloe_right_2.png*** với tham số đầu vào disparity_range = 64 và kernel_size = 5 ở cả hai hàm cost, ta được kết quả disparity map đã phần nào tệ đi (bị nhiễu) do ***Aloe_right_2.png*** có độ sáng hơn ***Aloe_right_1.png***...  
Độ đo L1 và L2 không có tính chất ***'invariant to linear changes'*** (bất biến với những thay đổi tuyến tính như cosine similarity và correlation coefficient). Do đó, L1 L2 sẽ không thể hoạt động tốt với hai ảnh cùng nội dung nhưng có một chút khác biệt liên quan đến độ sáng...

```
left_img_path = 'Aloe/Aloe_left_1.png'
right_img_path = 'Aloe/Aloe_right_2.png'
disparity_range = 64
kernel_size = 5

left = cv2.imread(left_img_path)
right = cv2.imread(right_img_path)

cv2_imshow(left)
cv2_imshow(right)
```
* **L1 Result:**

```
depth = window_based_matching_l1(
    left_img_path,
    right_img_path,
    disparity_range,
    kernel_size=kernel_size,
    save_result=False
)
cv2_imshow(depth)
cv2_imshow(cv2.applyColorMap(depth, cv2.COLORMAP_JET))
```
* **L2 Result:**

```
depth = window_based_matching_l2(
    left_img_path,
    right_img_path,
    disparity_range,
    kernel_size=kernel_size,
    save_result=False
)
cv2_imshow(depth)
cv2_imshow(cv2.applyColorMap(depth, cv2.COLORMAP_JET))
```

### **B_III. "invariant to linear changes" Solution**

1. Định nghĩa hàm **cosine_similarity():**

```
def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    return numerator / denominator
```

2. Định nghĩa hàm **window_based_matching():**

```
def window_based_matching(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    # Read left, right images then convert to grayscale
    left  = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3

    for y in range(kernel_half, height-kernel_half):
        for x in range(kernel_half, width-kernel_half):
            # Find j where cost has minimum value
            disparity = 0
            cost_optimal  = -1

            for j in range(disparity_range):
                d = x - j
                cost = -1
                if (d - kernel_half) > 0:
                    wp = left[(y-kernel_half):(y+kernel_half)+1, (x-kernel_half):(x+kernel_half)+1]
                    wqd = right[(y-kernel_half):(y+kernel_half)+1, (d-kernel_half):(d+kernel_half)+1]

                    wp_flattened = wp.flatten()
                    wqd_flattened = wqd.flatten()

                    cost = cosine_similarity(wp_flattened, wqd_flattened)

                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = j

            # Let depth at (y, x) = j (disparity)
            # Multiply by a scale factor for visualization purpose
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        # Save results
        cv2.imwrite('window_based_cosine_similarity.png', depth)
        cv2.imwrite('window_based_cosine_similarity_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth
```
**3. Disparity map Result:**

```
left_img_path = 'Aloe/Aloe_left_1.png'
right_img_path = 'Aloe/Aloe_right_2.png'
disparity_range = 64
kernel_size = 5

left = cv2.imread(left_img_path)
right = cv2.imread(right_img_path)

cv2_imshow(left)
cv2_imshow(right)
```
```
depth = window_based_matching(
    left_img_path,
    right_img_path,
    disparity_range,
    kernel_size=kernel_size,
    save_result=False
)
cv2_imshow(depth)
cv2_imshow(cv2.applyColorMap(depth, cv2.COLORMAP_JET))
```
