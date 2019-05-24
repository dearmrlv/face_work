from src import detect_faces, show_bboxes
from PIL import Image

image_1 = Image.open('test2.jpg')
bounding_boxes, landmarks = detect_faces(image_1)
image_2 = show_bboxes(image_1,bounding_boxes,landmarks)
image_2.show()
i = 0   # using for naming
for b in bounding_boxes:
    region = (b[0],b[1],b[2],b[3])
    face_cut = image_2.crop(region)
    face_cut = face_cut.resize((32,32))
    face_cut.show()
    face_cut.save('face_cut_'+str(i)+'.jpg')
    i += 1


