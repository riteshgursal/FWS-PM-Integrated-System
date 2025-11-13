**🏭 FWS-PM: Autonomous Factory Worker Safety & Predictive Maintenance**

This project is an integrated AI system for Smart Manufacturing, demonstrating the successful synthesis of multiple advanced deep learning concepts into a single, cohesive industrial solution.

It addresses critical industrial needs by linking human safety monitoring, AI model robustness, and machine health prediction.

**🚀 Knowledge Integration Showcase**

This integrated system demonstrates mastery across three specialized project domains:

The core function of the system is split between real-time monitoring and predictive analytics. The Worker Safety Check component showcases proficiency in (Real-Time CV) by simulating low-latency pose detection and alert generation near hazardous machinery. This capability is made robust and reliable through the (Generative AI) principles: we use synthetic data concepts to calculate a Robust Threshold, demonstrating an understanding of domain adaptation and model resilience to environmental challenges.

Simultaneously, the Predictive Maintenance component addresses (Harmonic Analysis). It uses time-series analysis techniques to analyze simulated vibration and harmonic features from the machine. We apply a Time-Series Classification model (Random Forest) to forecast the machine's health, ensuring continuous operation and preventing unplanned downtime.

**📁 Project Structure**

*fws-pm-integrated-system/**
├── main.py              # Executes the entire workflow (the entry point).
├── safety_pose.py       # Logic (Pose Detection & Synthetic Data Thresholding).
├── maintenance_ts.py    # Logic (Time-Series Data Generation & ML Classification).
├── requirements.txt
└── output/              # Stores visualization results (safety check images).


**🛠️ How to Run**

1. Setup

Clone the Repository and navigate into the fws-pm-integrated-system folder.

Create and Activate Environment (e.g., venv\Scripts\activate).

Install Dependencies:

pip install -r requirements.txt


2. Run the Integrated System

Execute the single main script to run all three demos sequentially:

python main.py


**Output**

The script will first print the Machine Health prediction and then execute the pose scenarios, generating two visual outputs in the output/ folder:

scenario_A_safe.png: (Worker Safe - Green Zone)

scenario_B_unsafe.png: (Worker Unsafe - Red Alert)

The console output will summarize the Machine Health Status and the Safety Alerts.
