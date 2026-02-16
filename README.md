<h1 align="center"> 
AI-Based-Tower-Component-Detection-Using-YOLOv8-and-YOLOv11
</h1>
<div align="justify">

<h3>
  Introduction
</h3>
Manual inspection of power infrastructure is labor-intensive and prone to errors. This project introduces a computer vision-based solution leveraging YOLOv8 and YOLOv11 to detect equipment such as transformers, insulators, circuit breakers, and potential hazards. Using CVAT for annotation, Ultralytics for training and deployment via Gradio, the pipeline is optimized for real-time and remote monitoring applications.
<br>
<br>
Traditional manual inspection methods are time-consuming, error-prone, and hazardous. This project presents an intelligent and automated solution for real-time object detection and fault analysis in power grid infrastructure using state-of-the-art deep learning models, specifically YOLOv8 and YOLOv11.
<br>
<br>
<h3>
  FUNCTIONAL REQUIREMENTS
</h3>
<p>
  
These define the core functionalities that the system must perform.

•	The system must process images of transmission towers and detect components such as:
<ul style="list-style-type:circle">  
      <li>Insulators</li>
      <li>	Grading-rings</li>
      <li>	CC-rings</li>
      <li>Tower parts</li>
      <li>	Anomalies like missing or reversed parts</li>
</ul>
•	It should allow annotation using CVAT and export data in YOLO format.<br>
•	The trained models (YOLOv8 and YOLOv11) should perform:
<ul style="list-style-type⚪">  
      <li>	Training with custom datasets.</li>
      <li>	Real-time or batch prediction/inference.</li>
      <li>Visualization of bounding boxes and class labels.</li>
</ul>
•	The system must support deployment via Gradio for interactive web-based inference.

•	Prediction results should include class name, confidence score, and bounding box coordinates.<br>
•	Optionally support image segmentation for precise boundary identification of objects.  

</p>

<h3>
  HARDWARE REQUIREMENTS
</h3>
<p >
  <table align="center">
  <tr>
    <th>Component</th>
    <th>Minimum Requirement</th>
    <th>Recommended</th>
  </tr>
  <tr>
    <td>CPU</td>
    <td>Intel i5</td>
    <td>Intel i7 or above</td>
  </tr>
  <tr>
    <td>RAM</td>
    <td>8 GB</td>
    <td>16 GB or above </td>
  </tr>
  <tr>
    <td>GPU</td>
    <td>GTX 1050 Ti (4GB)</td>
    <td>RTX 3060/3080 (6–12 GB VRAM)  </td>
  </tr>
  <tr>
    <td>Storage</td>
    <td>50 GB Free</td>
    <td> SSD with ≥ 100 GB </td>
  </tr>  
  
</table>
</p>
<h3>
  SOFTWARE REQUIREMENTS
</h3>
<P>
•	Operating System: Windows 10/11 or Ubuntu 20.04+<br>
•	Programming Language: Python 3.8+<br>
•	Libraries/Frameworks:<br>
	<ul style="list-style-type:circle">  
			o	PyTorch<br>
			o	Ultralytics YOLOv8/YOLOv11<br>
			o	OpenCV<br>
			o	NumPy, Matplotlib<br>
			o	Gradio<br>
			o	Eigen-CAM (for model explainability)<br>
	</ul>
•	Annotation Tool: CVAT (online or local deployment)<br>
•	Deployment Tools:<br>
<ul style="list-style-type:circle">  
	o	Flask (for REST API)<br>
	o	Docker (for containerization)<br>
	o	TensorRT/ONNX (for acceleration)<br>
</ul>	
</P>
<h3>
  DATASET REQUIREMENTS
</h3>
<p>
<ul style="list-style-type:circle"> 
•	A well-labelled dataset with classes such as:<br>
	<ul style="list-style-type:circle"> 
		o	grading-ring, grading-ring-missing, grading-ring-broken<br>
		o	insulator, cc-ring, tower-parts, no-class<br>
	</ul>
•	Annotation Format: YOLO format (.txt per image with class_id and normalized box)<br>
•	Dataset Split:<br>
		<ul style="list-style-type:circle"> 
			o	70% training<br>
			o	20% validation<br>
			o	10% test<br>
		</ul>
•	Augmentations:<br>
			<ul style="list-style-type:circle"> 
					o	Mosaic, MixUp, HSV jitter, flipping, brightness adjustments, etc.<br>
			</ul>
</ul>
	
</p>
<h3>
 TOOLS AND TECHNOLOGIES
</h3>
-	CVAT: Annotation tool for drawing bounding boxes and managing datasets.<br>
-	YOLOv8: A fast, anchor-free object detection model by Ultralytics.<br> 
<ul>
o	Augmentation: Augmentation involves transforming training data (e.g., flipping, rotating, 			scaling) to improve model generalization and robustness.<br> 	
</ul>	
-	YOLOv11: Upcoming architecture expected to integrate transformer components.<br>   
-	Python: Used for model scripting, training, and deployment APIs.<br>  
-	PyTorch: Backend framework for YOLOv8.<br>   
-	Ultralytics: Ultralytics is the organization behind YOLOv5 and YOLOv8, providing state-of-the-art, easy-to-use object detection models and tools.<br> 	
-	Eigen-CAM- Eigen-CAM is a class activation mapping technique that visualizes important
  regions in an image by leveraging the principal components of deep network activations.<br> 	
-	Gradio: Gradio is a Python library that lets you quickly create interactive web interfaces for machine learning models.<br> 


