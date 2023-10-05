import cv2
import matplotlib.pyplot as plt

cb_img = cv2.imread("../images/test.png", 0)
cb_img = cv2.resize(cb_img, (20, 20))

print(cb_img.size, cb_img.dtype)

plt.imshow(cb_img, cmap='gray')

plt.show()
