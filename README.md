## How to detect humans?

Create a conda enviorment with the following command:

```bash
conda create -n <env_name> python=3.8.18
```

Activate the enviorment:

```bash
conda activate <env_name>
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the main.py file:

```bash
python main.py
```

## Double Robot API inference
```bash
cd double
```
Make sure you have environment variable `PORT` set. 
```bash
# start socket connection
./setup.sh
```
Open new terminal window and run the following command:
```bash
python test_double.py
```
test_double.py should capture an image from the webcam and open it as a cv2 image.

## Organization

The project is organized in the following way:

- `main.py`: The main file of the project, where the main function is called.
- `LMRobot.py`: The class that represents the LM_robot. It has one method `detect_and_rank_humans` that returns the a ranked list of human descriptions based on each human's likelihood to assist the robot.
- `VLM.py`: All VLM class should extend `VLM` class. It has a method `inference` that takes in the image as a numpy array and the prompt as a string.
- `utils.py`: A file with some utility functions.
