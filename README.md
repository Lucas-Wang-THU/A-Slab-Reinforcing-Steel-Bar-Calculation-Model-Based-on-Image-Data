# A-Slab-Reinforcing-Steel-Bar-Calculation-Model-Based-on-Image-Data
This algorithm uses YOLOv11 to go through the top down image of the recognised layout rebar. Sweep recognition and YOLOv11 track mode are used to intelligently measure the rebar over a large area. Eventually, the algorithm will output geometric information as well as visualisation results.

The basic model is YOLOv11, and you can follow this project by using git clone https://github.com/meituan/YOLOv11.git. You can also access this exciting project via github!
You can build the env quikly by using pip:
pip install -r requirement.txt

Here is a description of the project. First, There are pre-trained weights files based on YOLOv11n.pt and YOLOv11s.pt in the wieghts folder. And the Yolov11_best_seg.pt's mIOU@0.5 is 98.9%. Secondly, the counter.py file for this project is an executable file which uses swipe recognition to identify large-scale images of plate reinforcement. You can set the img_path to counter your photo. The result will be saved in result folder.
