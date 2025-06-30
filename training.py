import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load your numerical data
# code for solar tracker model
data = pd.read_csv('C:\\Users\hp\\OneDrive\\Desktop\\crazy stuff\\solar_tracker_data.csv')
#df=pd.DataFrame(data)
X = data.drop(['Timestamp', 'Left', 'Right'], axis=1).to_numpy()
y = data[['ServoPosition',]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_classes = len(np.unique(y_train))

model_solar_tracker = keras.Sequential([
   layers.Dense(1000, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(128),
    layers.Dense(num_classes, activation='softmax')  
])
model_solar_tracker.compile(
    optimizer='adam',
    loss='mse',  
    metrics=['mae']  
)
history = model_solar_tracker.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  
    batch_size=32 
)
# Evaluate on test data
test_loss, test_metric = model_solar_tracker.evaluate(X_test, y_test)
print(f"Test performance: {test_metric}")
print(f"Test loss:{test_loss}")


predictions =model_solar_tracker.predict(X_test)
# model for the streets lights
# Dataset with high values meaning BRIGHT and low meaning DARK
data = [ 
    {"light_intensity": 93, "object_detected": 0, "led_state": 255},  # Dark + object → FULL
    {"light_intensity": 153, "object_detected": 0, "led_state": 255},
    {"light_intensity": 201, "object_detected": 0, "led_state": 255},
    
     {"light_intensity": 108, "object_detected": 0, "led_state": 255},  # Dark + object → FULL
    {"light_intensity": 159, "object_detected": 0, "led_state": 255},
    {"light_intensity": 202, "object_detected": 0, "led_state": 255},

     {"light_intensity": 90, "object_detected": 0, "led_state": 255},  # Dark + object → FULL
     {"light_intensity": 156, "object_detected": 0, "led_state": 255},
     {"light_intensity": 203, "object_detected": 0, "led_state": 255},

       {"light_intensity": 103, "object_detected": 0, "led_state": 255}, 
        {"light_intensity": 155, "object_detected": 0, "led_state": 255},
        {"light_intensity": 204, "object_detected": 0, "led_state": 255},

    {"light_intensity": 100, "object_detected": 0, "led_state": 255},  # Dark + object → FULL
    {"light_intensity": 150, "object_detected": 0, "led_state": 255},
    {"light_intensity": 205, "object_detected": 0, "led_state": 255},
    
    {"light_intensity": 100, "object_detected": 1, "led_state": 100},  # Dark + no object → DIM
    {"light_intensity": 150, "object_detected": 1, "led_state": 100},
    {"light_intensity": 200, "object_detected": 1, "led_state": 100},
    
    {"light_intensity": 900, "object_detected": 0, "led_state": 0},    # Bright + object → OFF
    {"light_intensity": 850, "object_detected": 0, "led_state": 0},
    {"light_intensity": 800, "object_detected": 0, "led_state": 0},
    
    {"light_intensity": 900, "object_detected": 1, "led_state": 0},    
    {"light_intensity": 850, "object_detected": 1, "led_state": 0},
    {"light_intensity": 800, "object_detected": 1, "led_state": 0},
    
    {"light_intensity": 500, "object_detected": 0, "led_state": 100}, 
    {"light_intensity": 550, "object_detected": 0, "led_state": 0},
    
    {"light_intensity": 501, "object_detected": 1, "led_state": 0}, 
    {"light_intensity": 552, "object_detected": 1, "led_state": 0},

    {"light_intensity": 700, "object_detected": 0, "led_state": 0}, 
    {"light_intensity": 550, "object_detected": 0, "led_state": 0},
    
    {"light_intensity": 503, "object_detected": 0, "led_state": 100}, 
    {"light_intensity": 551, "object_detected": 1, "led_state": 0},
    {"light_intensity": 598, "object_detected": 0, "led_state": 0}, 
    {"light_intensity": 554, "object_detected": 0, "led_state": 0},
    
    {"light_intensity": 580, "object_detected": 1, "led_state": 0}, 
    {"light_intensity": 559, "object_detected": 0, "led_state": 0},

    {"light_intensity": 400, "object_detected": 0, "led_state": 255}, 
    {"light_intensity": 750, "object_detected": 0, "led_state": 0},
    
    {"light_intensity": 300, "object_detected": 1, "led_state": 100}, 
    {"light_intensity": 844, "object_detected": 1, "led_state": 0},


]


df = pd.DataFrame(data)

# Features and labels
X = df[["light_intensity", "object_detected"]]
y = df["led_state"]

# Train the model
model_lights = KNeighborsClassifier(n_neighbors=3)
model_lights.fit(X, y)

# Predict new input
new_light_intensity = 100  # low light
new_object_detected = 0  # object prese


predicted_led = model_lights.predict([[new_light_intensity, new_object_detected]])[0]

# Output
print("INPUT:")
print(f"  Light intensity: {new_light_intensity} (higher = brighter)")
print(f"  Object detected: {'Yes' if new_object_detected == 0 else 'No'}")
print(f"PREDICTED LED STATE: {predicted_led}")
