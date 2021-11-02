import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog as fd
import dlib

class Assignment2(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, ImagePage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Assignment 2", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="Morph Page",
                            command=lambda: controller.show_frame("ImagePage"))
        button1.pack()

class ImagePage(tk.Frame):

    panel1 = None
    panel2 = None
    panel3 = None
    image1 = None
    image2 = None
    image1_formorph = None
    image2_formorph = None
    featurePoints1 = None
    featurePoints2 = None
    featurePoints1_formorph = None
    featurePoints2_formorph = None

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Image Page", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: self.go_to_start_page())
        button.pack()
        button1 = tk.Button(self, text="Load Image 1", command=lambda: self.load_image1())
        button1.pack()
        button2 = tk.Button(self, text="Load Image 2", command=lambda: self.load_image2())
        button2.pack()
        button7 = tk.Button(self, text="Morph Images", command=lambda: self.image_morph())
        button7.pack()
        button3 = tk.Button(self, text="Show Feature points Image 1", command=lambda: self.feature_detection1())
        button3.pack()
        button4 = tk.Button(self, text="Show Feature points Image 2", command=lambda: self.feature_detection2())
        button4.pack()
        button5 = tk.Button(self, text="Show Delauney Triangulation Image 1", command=lambda: self.delauney_triangulation1())
        button5.pack()
        button6 = tk.Button(self, text="Show Delauney Triangulation Image 2", command=lambda: self.delauney_triangulation2())
        button6.pack()


    def go_to_start_page(self):

        if self.panel1 is not None or self.panel2 is not None:
            self.panel1.pack_forget()
            self.panel2.pack_forget()

        self.controller.show_frame("StartPage")

    def load_image1(self):

        path = fd.askopenfilename()

        if len(path) > 1:
            image1 = cv2.imread(path)
            self.image1 = image1
            self.image1_formorph = image1
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = Image.fromarray(image1)
            image1 = ImageTk.PhotoImage(image1)

            self.panel1 = tk.Label(image=image1)
            self.panel1.image = image1
            self.panel1.pack(side="left", padx=10, pady=10)
            self.panel1.configure(image=image1)
            self.panel1.image = image1

    def load_image2(self):

        path = fd.askopenfilename()

        if len(path) > 1:
            image2 = cv2.imread(path)
            self.image2 = image2
            self.image2_formorph = image2
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = Image.fromarray(image2)
            image2 = ImageTk.PhotoImage(image2)

            self.panel2 = tk.Label(image=image2)
            self.panel2.image = image2
            self.panel2.pack(side="left", padx=10, pady=10)
            self.panel2.configure(image=image2)
            self.panel2.image = image2

    def feature_detection1(self):
        def shape_to_numpy_array(shape, dtype="int"):
            coordinates = np.zeros((80, 2), dtype=dtype)

            for i in range(0, 68):

                coordinates[i] = (shape.part(i).x, shape.part(i).y)

                x = [538, 215, 0, 599, 0, 0, 0, 300, 599, 599, 599, 300]
                y = [410, 727, 760, 597, 0, 400, 799, 799, 799, 400, 0, 0]

                x = np.array(x)
                y = np.array(y)

                for i in range(len(x)):
                    coordinates[i + 67][0] = x[i]
                    coordinates[i + 67][1] = y[i]

            return coordinates

        def draw_point(img, p, color):
            cv2.circle(img, p, 2, color, 0)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        image = self.image1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = shape_to_numpy_array(shape)

        np.savetxt("feature_points1.txt", shape, fmt='%.0f')
        self.featurePoints1 = "feature_points1.txt"

        points = [];
        with open(self.featurePoints1) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))

        for p in points:
            draw_point(image, p, (255, 0, 235))

        cv2.imshow('feature_detection1', image)

    def feature_detection2(self):
        def shape_to_numpy_array(shape, dtype="int"):
            coordinates = np.zeros((80, 2), dtype=dtype)

            for i in range(0, 68):

                coordinates[i] = (shape.part(i).x, shape.part(i).y)

                x = [538, 215, 0, 599, 0, 0, 0, 300, 599, 599, 599, 300]
                y = [410, 727, 760, 597, 0, 400, 799, 799, 799, 400, 0, 0]

                x = np.array(x)
                y = np.array(y)

                for i in range(len(x)):
                    coordinates[i + 67][0] = x[i]
                    coordinates[i + 67][1] = y[i]

            return coordinates

        def draw_point(img, p, color):
            cv2.circle(img, p, 2, color, 0)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        image = self.image2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = shape_to_numpy_array(shape)

        np.savetxt("feature_points2.txt", shape, fmt='%.0f')
        self.featurePoints2 = "feature_points2.txt"

        points = [];
        with open(self.featurePoints2) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))

        for p in points:
            draw_point(image, p, (255, 0, 235))

        cv2.imshow('feature_detection2', image)

    def delauney_triangulation1(self):
        def rect_contains(rect, point):
            if point[0] < rect[0]:
                return False
            elif point[1] < rect[1]:
                return False
            elif point[0] > rect[2]:
                return False
            elif point[1] > rect[3]:
                return False
            return True

        # Draw a point
        def draw_point(img, p, color):
            cv2.circle(img, p, 2, color, 0)

        # Draw delaunay triangles
        def draw_delaunay(img, subdiv, delaunay_color):
            triangleList = subdiv.getTriangleList();
            size = img.shape
            r = (0, 0, size[1], size[0])

            for t in triangleList:

                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
                    cv2.line(img, pt1, pt2, delaunay_color, 1, 0)
                    cv2.line(img, pt2, pt3, delaunay_color, 1, 0)
                    cv2.line(img, pt3, pt1, delaunay_color, 1, 0)

        # Define window names
        win_delaunay = "Delaunay Triangulation"

        # Define colors for drawing.
        delaunay_color = (255, 255, 255)
        points_color = (0, 0, 255)

        # Read in the image.
        img = self.image1

        # Keep a copy around
        img_orig = img.copy();

        # Rectangle to be used with Subdiv2D
        size = img.shape
        rect = (0, 0, size[1], size[0])

        # Create an instance of Subdiv2D
        subdiv = cv2.Subdiv2D(rect);

        # Create an array of points.
        points = [];

        # Read in the points from a text file
        with open(self.featurePoints1) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))

        # Insert points into subdiv
        for p in points:
            subdiv.insert(p)

        # Draw delaunay triangles
        draw_delaunay(img, subdiv, (255, 255, 255));

        # Draw points
        for p in points:
            draw_point(img, p, (0, 0, 255))

        # Show results
        cv2.imshow(win_delaunay, img)

    def delauney_triangulation2(self):
        def rect_contains(rect, point):
            if point[0] < rect[0]:
                return False
            elif point[1] < rect[1]:
                return False
            elif point[0] > rect[2]:
                return False
            elif point[1] > rect[3]:
                return False
            return True

        # Draw a point
        def draw_point(img, p, color):
            cv2.circle(img, p, 2, color, 0)

        # Draw delaunay triangles
        def draw_delaunay(img, subdiv, delaunay_color):
            triangleList = subdiv.getTriangleList();
            size = img.shape
            r = (0, 0, size[1], size[0])

            for t in triangleList:

                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
                    cv2.line(img, pt1, pt2, delaunay_color, 1, 0)
                    cv2.line(img, pt2, pt3, delaunay_color, 1, 0)
                    cv2.line(img, pt3, pt1, delaunay_color, 1, 0)

        win_delaunay = "Delaunay Triangulation"

        # Define colors for drawing.
        delaunay_color = (255, 255, 255)
        points_color = (0, 0, 255)

        # Read in the image.
        img = self.image2

        # Keep a copy around
        img_orig = img.copy();

        # Rectangle to be used with Subdiv2D
        size = img.shape
        rect = (0, 0, size[1], size[0])

        # Create an instance of Subdiv2D
        subdiv = cv2.Subdiv2D(rect);

        # Create an array of points.
        points = [];

        # Read in the points from a text file
        with open(self.featurePoints2) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))

        # Insert points into subdiv
        for p in points:
            subdiv.insert(p)

        # Draw delaunay triangles
        draw_delaunay(img, subdiv, (255, 255, 255));

        # Draw points
        for p in points:
            draw_point(img, p, (0, 0, 255))

        # Show results
        cv2.imshow(win_delaunay, img)

    def image_morph(self):

        def readPoints(path):
            # Create an array of points.
            points = [];
            # Read points
            with open(path) as file:
                for line in file:
                    x, y = line.split()
                    points.append((int(x), int(y)))

            return points

        # Apply affine transform calculated using srcTri and dstTri to src and
        # output an image of size.
        def applyAffineTransform(src, srcTri, dstTri, size):
            # Given a pair of triangles, find the affine transform.
            warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

            # Apply the Affine Transform just found to the src image
            dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

            return dst

        # Warps and alpha blends triangular regions from img1 and img2 to img
        def morphTriangle(img1, img2, img, t1, t2, t, alpha):
            # Find bounding rectangle for each triangle
            r1 = cv2.boundingRect(np.float32([t1]))
            r2 = cv2.boundingRect(np.float32([t2]))
            r = cv2.boundingRect(np.float32([t]))

            # Offset points by left top corner of the respective rectangles
            t1Rect = []
            t2Rect = []
            tRect = []

            for i in range(0, 3):
                tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
                t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
                t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

            # Get mask by filling triangle
            mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

            # Apply warpImage to small rectangular patches
            img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
            img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

            size = (r[2], r[3])
            warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
            warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

            # Alpha blend rectangular patches
            imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

            # Copy triangular region of the rectangular patch to the output image
            img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (
                        1 - mask) + imgRect * mask

        def get_delaunay_indexes(img1, points):

            rect = (0, 0, img1.shape[1], img2.shape[0])
            subdiv = cv2.Subdiv2D(rect);
            for p in points:
                subdiv.insert(p)

            triangleList = subdiv.getTriangleList();
            triangles = []
            for p in triangleList:
                vertexes = [0, 0, 0]
                for v in range(3):
                    vv = v * 2
                    for i in range(len(points)):
                        if p[vv] == points[i][0] and p[vv + 1] == points[i][1]:
                            vertexes[v] = i

                triangles.append(vertexes)

            return triangles

        def feature_detection1(image):
            def shape_to_numpy_array(shape, dtype="int"):
                coordinates = np.zeros((80, 2), dtype=dtype)

                for i in range(0, 68):
                    coordinates[i] = (shape.part(i).x, shape.part(i).y)

                x = [538, 215, 0, 599, 0, 0, 0, 300, 599, 599, 599, 300]
                y = [410, 727, 760, 597, 0, 400, 799, 799, 799, 400, 0, 0]

                x = np.array(x)
                y = np.array(y)

                for i in range(len(x)):
                    coordinates[i+67][0] = x[i]
                    coordinates[i+67][1] = y[i]

                return coordinates

            def draw_point(img, p, color):
                cv2.circle(img, p, 2, color, 0)

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = shape_to_numpy_array(shape)

            np.savetxt("feature_points1_formorph.txt", shape, fmt='%.0f')
            self.featurePoints1_formorph = "feature_points1_formorph.txt"

            points = [];
            with open(self.featurePoints1_formorph) as file:
                for line in file:
                    x, y = line.split()
                    points.append((int(x), int(y)))

            for p in points:
                draw_point(image, p, (255, 0, 235))

        def feature_detection2(image):
            def shape_to_numpy_array(shape, dtype="int"):
                coordinates = np.zeros((80, 2), dtype=dtype)

                for i in range(0, 68):
                    coordinates[i] = (shape.part(i).x, shape.part(i).y)

                x = [538, 215, 0, 599, 0, 0, 0, 300, 599, 599, 599, 300]
                y = [410, 727, 760, 597, 0, 400, 799, 799, 799, 400, 0, 0]

                x = np.array(x)
                y = np.array(y)

                for i in range(len(x)):
                    coordinates[i + 67][0] = x[i]
                    coordinates[i + 67][1] = y[i]

                return coordinates

            def draw_point(img, p, color):
                cv2.circle(img, p, 2, color, 0)

            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = shape_to_numpy_array(shape)

            np.savetxt("feature_points2_formorph.txt", shape, fmt='%.0f')
            self.featurePoints2_formorph = "feature_points2_formorph.txt"

            points = [];
            with open(self.featurePoints2_formorph) as file:
                for line in file:
                    x, y = line.split()
                    points.append((int(x), int(y)))

            for p in points:
                draw_point(image, p, (255, 0, 235))

        alpha = 0.5

        # Read images
        img1 = self.image1_formorph
        img2 = self.image2_formorph

        # Convert Mat to float data type
        img1 = np.float32(img1)
        img2 = np.float32(img2)

        feature_detection1(self.image1_formorph)
        feature_detection2(self.image2_formorph)

        print(self.featurePoints1)

        # Read array of corresponding points
        points1 = readPoints(self.featurePoints1_formorph)
        points2 = readPoints(self.featurePoints2_formorph)
        points = [];

        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))

        # Allocate space for final output
        imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

        delaunay = get_delaunay_indexes(img1, points1)

        for v1, v2, v3 in delaunay:
            t1 = [points1[v1], points1[v2], points1[v3]]
            t2 = [points2[v1], points2[v2], points2[v3]]
            t = [points[v1], points[v2], points[v3]]

                #Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

        # Display Result
        cv2.imshow("Morphed Face", np.uint8(imgMorph))
        cv2.waitKey(0)

if __name__ == "__main__":
    app = Assignment2()
    app.mainloop()