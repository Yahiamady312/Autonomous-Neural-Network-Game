# 🧠 Autonomous Game-Playing Neural Network

This project combines a **2D lunar lander-style game** with a **neural network** that learns to play it automatically.  
It is based on the *CE889 – Neural Networks and Deep Learning* lab assignment *“Data Collection and Game Automation”*.  

---

## 🎯 Project Overview

The project has two main parts:

1. **The Game**  
   A simple simulation where a lander must reach a target without crashing.  
   The player (or later, the neural network) applies thrust to control the craft’s position and velocity.

2. **The Neural Network**  
   A Python model trained on data collected from human gameplay.  
   It learns to predict the proper thrust controls from the lander’s position and velocity.

---

## 🧩 How It Works

### 1. Play the Game and Collect Data
- Run the game locally.  
- Every session records one or more game runs to a CSV file (e.g. `ce889_dataCollection.csv`).  
- Each row contains one frame of gameplay.

**Inputs (state variables)**  
| Variable | Type | Description |
|-----------|------|-------------|
| X distance to target | Float | Horizontal distance (pixels) |
| Y distance to target | Float | Vertical distance (pixels) |
| Velocity X | Float | Horizontal speed (pixels / s) |
| Velocity Y | Float | Vertical speed (pixels / s) |

**Outputs (control actions)**  
| Variable | Type | Description |
|-----------|------|-------------|
| Thrust / Movement | Int / Float | Control command applied at that frame |

---

### 2. Prepare the Data
Before training the neural network:
1. **Normalise** data to a 0 – 1 range.  
2. **Clean** inconsistent or missing rows.  
3. **Split** into training and validation sets.  

---

### 3. Train the Neural Network
The script `Neural network.py`:
- Reads the CSV file from the game.  
- Builds a feed-forward neural network.  
- Trains it to map game state → control actions.  
- Saves the model weights for autonomous play.

---

## 📁 Folder Structure

```text
Autonomous Neural Network Game/
├── Game/                   # Game engine, assets, CSV generation
├── Neural network.py        # Neural-network training code
├── .gitignore               # Ignore venv, cache, and temporary files
└── README.md                # Project documentation
```

---

## ⚙️ How to Run

1. **Set up environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   pip install -r Game/requirements.txt
   ```

2. **Run the game**
   ```bash
   python Game/Main.py
   ```

3. **Play and close the game** — data will appear in your CSV file.

4. **Train the neural network**
   ```bash
   python "Neural network.py"
   ```

---

## 🧠 Learning Outcome

This project demonstrates:
- Collecting real-time sensor data from gameplay.  
- Pre-processing and normalising numeric datasets.  
- Designing and training a neural network to mimic human decisions.  
- Applying AI control to interactive systems.

---

## 🧰 Requirements
- Python 3.9 +
- Pygame (for the game)
- NumPy, Pandas, TensorFlow / PyTorch (for NN)

---

## 👨‍💻 Author
**Yahia Mady**  
University of Essex – CE889 Neural Networks and Deep Learning  
