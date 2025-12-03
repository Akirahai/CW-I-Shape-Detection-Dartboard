import cv2

coords = []

def draw_box(event, x, y, flags, param):
    global coords, img
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        coords.append((x, y))
        cv2.rectangle(img, coords[0], coords[1], (0, 255, 0), 2)
        cv2.imshow('image', img)
        # print the coordinates x,y, width, height
        x1, y1 = coords[0]
        width = coords[1][0] - x1
        height = coords[1][1] - y1
        properties = (x1, y1, width, height)
        print("Bounding box:", properties)

img = cv2.imread('Dartboard/dart15.jpg')
img_original = img.copy()
cv2.namedWindow('image')        # important
cv2.imshow('image', img)
cv2.setMouseCallback('image', draw_box)

# Wait until window is closed
while True:
    key = cv2.waitKey(20)

    # if user presses ESC
    if key == 27:
        break
    
    # Press '1' to reset image
    if key == ord('1'):
        img = img_original.copy()
        cv2.imshow('image', img)
        print("Image reset") 

    # if the window is closed
    if cv2.getWindowProperty('image', cv2.WND_PROP_AUTOSIZE) < 0:
        break

cv2.destroyAllWindows()
