import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image, ImageEnhance
import warnings
from sklearn.utils.multiclass import unique_labels