## Common Coordinate Behaviour: arena alignment and fisheye correction
**Code to generate and apply a geometric transform to align videos that may be at different positions, angles, and zoom. We use this to analyze escape trajectories from different sessions within the same coordinate space. Includes code to correct for fisheye lens distortion.**

Use _Register_and_Display_Behavior.py_ to register videos and display behavioral events

Relevant functions are found in _Video_Functions.py_. In particular the _model_arena()_ function must be customized to match the particular arena or object in your videos.

Use _Calibrate_Fisheye.py_ to calibrate fisheye distortion in your camera

To test it out, run the file _Register_and_Display_Behavior.py_

Lastly, be sure to first install open-cv! (in my case, I pip installed a file called opencv_python-3.4.0+contrib-cp36-cp36m-win_amd64 downloaded from https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)

![](https://github.com/BrancoLab/Common-Coordinate-Behaviour/blob/master/example.JPG)
The GUI from the _register_arena()_ function. Clicking on the points on the video frame (left) that correspond to the four points indicated in the model arena (center) transforms the image such that it is aligned and overlaid with the model arena (right)

![](https://github.com/BrancoLab/Common-Coordinate-Behaviour/blob/master/example2.jpg)

Example output: escape trajectories plotted within the same coordinate space
