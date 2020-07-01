import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from time import sleep # Import the sleep function from the time module
greenLEDPin = 8
redLEDPin = 7

def setup():
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    # Set pin greenLEDPin to be an output pin and set initial value to low (off)
    GPIO.setup(greenLEDPin, GPIO.OUT, initial=GPIO.LOW) 
    # Set pin redLEDPin to be an output pin and set initial value to low (off)
    GPIO.setup(redLEDPin, GPIO.OUT, initial=GPIO.LOW) 
   

def activateGreenLED():
    GPIO.output(greenLEDPin, GPIO.HIGH) # Turn on
    GPIO.output(redLEDPin, GPIO.LOW) # Turn off

def activateRedLED():
    GPIO.output(greenLEDPin, GPIO.LOW) # Turn off
    GPIO.output(redLEDPin, GPIO.HIGH) # Turn on

def deactivateAll():
    GPIO.output(greenLEDPin, GPIO.LOW) # Turn off
    GPIO.output(redLEDPin, GPIO.LOW) # Turn off