from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

from pylab import *
import peakutils
from peakutils.plot import plot as pplot
from matplotlib import pyplot
from scipy import signal
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas
import pygame
import ctypes
import serial
import sys

import speech_recognition as sr
import pyttsx

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                   pygame.color.THECOLORS["blue"], 
                   pygame.color.THECOLORS["green"], 
                   pygame.color.THECOLORS["orange"], 
                   pygame.color.THECOLORS["purple"], 
                   pygame.color.THECOLORS["yellow"], 
                   pygame.color.THECOLORS["violet"]]

class RobotRuntime():
    
    def __init__(self):
    
        self.listener = sr.Recognizer()
        self.speaker = pyttsx.init()
        self.speaker.setProperty('rate', 150)
        
        self.voices = self.speaker.getProperty('voices')
        self.speaker.setProperty('voice', self.voices[1].id)
        
        self.ser = serial.Serial(port='COM3', baudrate=9600)
        
        pygame.init()

        # Used to manage how fast the screen updates
        self.clock = pygame.time.Clock()
        self.time = pygame.time
        self.current_time = 0

        # Set the width and height of the screen [width, height]
        self.info_object = pygame.display.Info()
        self.screen = pygame.display.set_mode((self.info_object.current_w >> 1, self.info_object.current_h >> 1), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Vitals Robot")

        # Loop until the user clicks the close button.
        self.done = False

        # Kinect runtime object, we want only color and body frames 
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self.frame_surface = pygame.Surface((self.kinect.color_frame_desc.Width, self.kinect.color_frame_desc.Height), 0, 32)

        # store skeleton data 
        self.bodies = None
        
        # patient name
        self.patient_name = None
        
        # extra wait flag for bp/pulse
        self.bp_wait = False

    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
    
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self.frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
            
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self.kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()
        
    def audio_input(self):
    
        audio_string = None
        
        with sr.Microphone() as source:
            print("Speak:")
            audio = self.listener.listen(source)
            try:
                audio_string = self.listener.recognize_google(audio)
                print("You said " + audio_string)
            except sr.UnknownValueError:
                self.speak("Sorry! I couldn't understand what you said.")
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
        
        return audio_string
        
    def speak(self, input):
        self.speaker.say(input)
        self.speaker.runAndWait()

    def run(self):
    
        self.speak("""Hello! Please sit in front of the Kinect camera and move your torso for about 1 second until I detect you. 
                      Try raising your arms if I seem to not be able to detect you.""")
    
        while not self.done:
        
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self.done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self.screen = pygame.display.set_mode(event.dict['size'], 
                                  pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            # --- Getting frames and drawing
            if self.kinect.has_new_color_frame():
                frame = self.kinect.get_last_color_frame()
                self.draw_color_frame(frame, self.frame_surface)
                frame = None

            # --- We have a body frame, so can get skeletons
            if self.kinect.has_new_body_frame(): 
                self.bodies = self.kinect.get_last_body_frame()
                
            # --- draw skeletons to frame_surface
            if self.bodies is not None:
            
                for i in range(0, self.kinect.max_body_count):
                    body = self.bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints
                    
                    # convert joint coordinates to color space 
                    joint_points = self.kinect.body_joints_to_color_space(joints)
                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])
                    
                    if not taskList["Task1"][1]: # want measurement?
                        
                        self.speak(taskList["Task1"][0])
                        audio_string = self.audio_input()
                        
                        if "yes" in audio_string:
                            taskList["Task1"][1] = True
                            
                    if taskList["Task1"][1] and not taskList["Task2"][1]: # ask name
                        
                        self.speak(taskList["Task2"][0])
                        audio_string = self.audio_input()
                        
                        if audio_string is not None:
                            self.patient_name = audio_string
                            taskList["Task2"][1] = True
                    
                    if taskList["Task2"][1] and not taskList["Task3"][1]: # temperature instructions
                        
                        self.speak(taskList["Task3"][0].format(self.patient_name))
                        
                        taskList["Task3"][1] = True
                        self.current_time = self.time.get_ticks()
                        
                    if taskList["Task3"][1] and not taskList["Task4"][1]: # temperature measurement
                        if self.time.get_ticks() < (self.current_time + 120000): # and self.ser.isOpen(): # track for 2 minutes
                        
                            temperature = float(self.ser.readline().decode('utf-8')) + float(2.0)
                            print("Temperature: " + str(temperature))
                            
                            # Respiration rate
                            distance.append(joints[1].Position.z)
                            milliseconds.append(self.time.get_ticks())
                        
                        else:
                            taskList["Task4"][1] = True
                            self.current_time = self.time.get_ticks()
                    
                    if taskList["Task4"][1] and not taskList["Task5"][1]: # bp/pulse instructions
                        if self.time.get_ticks() < (self.current_time + 3000): # instructions only once, allow 3 seconds
                        
                            self.speak(taskList["Task5"][0])
                        
                        elif self.time.get_ticks() < (self.current_time + 120000): # wait for 2 minutes
                            pass
                        else:
                            taskList["Task5"][1] = True
                            self.current_time = self.time.get_ticks()
                            
                    if taskList["Task5"][1] and not taskList["Task6"][1]:
                        if self.time.get_ticks() < (self.current_time + 3000) and not self.bp_wait: # give more time for bp/pulse
                    
                            self.speak(taskList["Task6"][0])
                            audio_string = self.audio_input()
                            
                            if "yes" in audio_string:
                                taskList["Task6"][1] = True
                            else:
                                self.speak("Okay! I will give you another minute.")
                                self.bp_wait = True
                        elif self.time.get_ticks() < (self.current_time + 60000) and self.bp_wait: # wait for 1 minute
                            pass
                        else:
                        
                            self.speak(taskList["Task6"][0])
                            audio_string = self.audio_input()
                            
                            if "yes" in audio_string:
                                taskList["Task6"][1] = True
                            else:
                                self.speak("Okay! I will give you another minute.")
                                self.current_time = self.time.get_ticks()
                            
                    if taskList["Task6"][1] and not taskList["Task7"][1]: # systolic input
                        
                        self.speak(taskList["Task7"][0])
                        audio_string = self.audio_input()
                        
                        if audio_string is not None:
                            try:
                                systolic = float(audio_string)
                            except ValueError:
                                self.speak("You must tell me a number!")
                            else:
                                taskList["Task7"][1] = True
                    
                    if taskList["Task7"][1] and not taskList["Task8"][1]: # diastolic input
                        
                        self.speak(taskList["Task8"][0])
                        audio_string = self.audio_input()
                        
                        if audio_string is not None:
                            try:
                                diastolic = float(audio_string)
                            except ValueError:
                                self.speak("You must tell me a number!")
                            else:
                                taskList["Task8"][1] = True
                            
                    if taskList["Task8"][1] and not taskList["Task9"][1]: # pulse input
                        
                        self.speak(taskList["Task9"][0])
                        audio_string = self.audio_input()
                        
                        if audio_string is not None:
                            try:
                                pulse = float(audio_string)
                            except ValueError:
                                self.speak("You must tell me a number!")
                            else:
                                taskList["Task9"][1] = True
                            
                    if taskList["Task9"][1] and not taskList["Task10"][1]: # want to learn results?
                        
                        self.speak(taskList["Task10"][0])
                        audio_string = self.audio_input()
                        
                        if "yes" in audio_string:
                            taskList["Task10"][1] = True
                            self.done = True
                        elif "no" in audio_string:
                            self.speak("Okay! Goodbye and take care!")
                            sys.exit()
                        else:
                            pass

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size) 
            h_to_w = float(self.frame_surface.get_height()) / self.frame_surface.get_width()
            target_height = int(h_to_w * self.screen.get_width())
            surface_to_draw = pygame.transform.scale(self.frame_surface, (self.screen.get_width(), target_height));
            self.screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 30 frames per second
            self.clock.tick(30)
            # if robot.step(timestep) == -1:
        self.kinect.close()
        pygame.quit()

taskList = {"Task1": ["Great! I can see you now. Would you like your vital signs collected?", False], 
            "Task2": ["What is your name?", False],
            "Task3": ["""Thank you, {}! First, I will take your temperature. 
                         Please tightly grip the black temperature sensor located on the green board with your index finger and your thumb supporting the sensor underneath.
                         Hold your fingers on the sensor for about 2 minutes and remember to sit still and do not slouch while the measurement is being taken.
                         Fidgeting or moving will disrupt your results. I will start measuring your temperature as soon as you grip the sensor.""", False],
            "Task4": ["Temperature sensor", False],
            "Task5": ["""Thank you! Next, you will help me take your blood pressure and pulse measurement. I will tell you everything you need to know.
                         Please start by placing the blood pressure cuff in front of you around your left bicep, approximately 1 inch above the crease of your elbow.
                         Ensure that the black cable attached to the cuff is also attached to the receiver.
                         Tighten the cuff so that it fits snuggly around your bicep without being uncomfortably tight.
                         Power on the receiver and press the green button to start the measurement.
                         Sit still until the measurement is complete. I will check back with you in 2 minutes.""", False],
            "Task6": ["Are you ready to proceed with the measurement readings?", False],
            "Task7": ["Please read the systolic blood pressure on the screen. It will be the largest number.", False],
            "Task8": ["Please read the diastolic blood pressure on the screen. It is located underneath the systolic blood pressure.", False],
            "Task9": ["Please read the pulse number. It is located on the bottom right hand side of the screen", False],
            "Task10": ["Thank you for the information! Would you like your results to be explained?", False],
            "Task11": ["It looks like you might have {}. Please discuss any further questions with the doctor. Thank you and take care!", False]}

distance = []
milliseconds = []

diastolic = 0
systolic = 0
pulse = 0
temperature = 0

url = "C:\\Path\\to\\file\\vitals.csv" # Change this
names = ['Diastolic BP', 'Heart Rate', 'Respiratory Rate', 'Systolic BP', 'Temperature', 'DIAGNOSIS']
dataset = pandas.read_csv(url, names=names)

dataset = dataset.sample(frac=1).reset_index(drop=True)
array = dataset.values

x_axis = array[:, 0:5]
y_axis = array[:, 5]

runtime = RobotRuntime()
runtime.run()

N  = 3 # Filter order
Wn = 0.1 # Cutoff frequency
B, A = signal.butter(N, Wn)

try:
    smooth_distance = signal.filtfilt(B, A, distance)
    smooth_distance_array = np.array(smooth_distance)
    indexes = peakutils.indexes(smooth_distance_array)
    
    # pyplot.figure()
    # pplot(np.array(milliseconds), smooth_distance_array, indexes)
    # pyplot.plot(milliseconds, distance, 'y-')
    # pyplot.title('Respiration Rate')
    # pyplot.ylabel('Distance (m)')
    # pyplot.xlabel('milliseconds')
    # pyplot.show()
except ValueError:
    pass #print("Insufficient respiration rate data!")
    
x_validation = np.asarray([diastolic, pulse, len(indexes), systolic, temperature]).reshape(1, -1) # Diastolic BP, Heart Rate, Respiratory Rate, Systolic BP, Temperature

cart = DecisionTreeClassifier()
cart.fit(x_axis, y_axis)
prediction = cart.predict(x_validation)

print(prediction)
runtime.speak(taskList["Task11"][0].format(prediction))
