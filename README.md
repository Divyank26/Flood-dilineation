Abstract
Floods are among the most destructive natural disasters worldwide. Traditional flood mapping methods are time-consuming and heavily weather-dependent. Sentinel-1 Synthetic Aperture Radar (SAR) imagery overcomes this by enabling continuous flood monitoring regardless of cloud cover or lighting conditions. This project develops a two-stage AI pipeline featuring a CNN-based Flood Presence Classifier and an Attention U-Net Flood Segmentation model. The system integrates flood statistics, severity estimation, and recovery recommendations, all deployed through an interactive Streamlit web application.

Introduction & Problem Statement
Manual flood assessment is slow and unsuitable for rapid disaster response. Existing systems often rely on optical satellite imagery, which is frequently obstructed by cloud cover during active flood events. There is a critical need for an automated flood detection and mapping system capable of operating under all weather conditions.
Sentinel-1 SAR Imagery offers a robust solution. Using C-band Synthetic Aperture Radar, it operates day and night, easily penetrates clouds, provides 10 m spatial resolution, and utilizes dual polarization (VV and VH) to accurately capture water bodies.

Objectives
Detect whether flooding exists in a given satellite image.
Segment the specific flooded regions accurately.
Estimate the total flood area and calculate percentage coverage.
Determine flood severity based on established thresholds.
Generate actionable recovery recommendations.

Dataset Details
The model leverages the Sen1Floods11 dataset.
Feature       Description              Preprocessing Steps
Imagery       Sentinel-1 SAR Images    Read GeoTIFF
Labels        LabelHand flood masks    Normalize VV and VH channels
Polarization  Dual (VV + VH)           Resize to 256×256 pixels
Dimensions    Originally 512×512       Generate binary flood labels
Volume        4384 image-mask pairs    Store processed dataset

System Architecture 
<img width="558" height="527" alt="Screenshot 2026-07-25 140712" src="https://github.com/user-attachments/assets/06327831-8596-493c-b7ee-f40856d76119" />



Methodology
1. Flood Presence Classification:
A Convolutional Neural Network (CNN) determines if the VV and VH input image contains flooded regions. It outputs a binary result (Flood / No Flood) using BCEWithLogitsLoss. Confidence calibration is applied post-training via temperature scaling.
2. Flood Segmentation:
Images flagged for flooding pass to an Attention U-Net model (2-channel tensor input). Utilizing an Encoder/Decoder structure, skip connections, and attention gates, it outputs a precise binary flood mask.
4. Flood Statistics & Severity Estimation:
   Flood Coverage: Flood Pixels / Total Pixels
   Flood Area: Flood Pixels × 100 m²
   Flood Coverage         Severity Level
   < 5%                      Low
   5 – 20%                   Moderate
   20 – 40%                  High
   > 40%                     Severe
5. Recovery Recommendation:
Based on the computed severity level and coverage, the system generates automated guidelines including emergency response protocols, infrastructure inspection plans, evacuation notices, and rehabilitation strategies.

Model Training & Evaluation Metrics
Both models were trained using the Adam Optimizer. The CNN utilized BCEWithLogitsLoss and batch processing, while the Attention U-Net utilized Dice Loss alongside model checkpointing based on validation loss.
Metric            Score
Mean IoU          0.7947
Mean Dice         0.8628
Precision         0.8836
Recall            0.8973
Pixel Accuracy    0.9882

Experimental Results & Deployment
The system is deployed via a Streamlit Dashboard providing a user-friendly, end-to-end workflow:
Upload TIFF ➔ Preprocessing ➔ Classification ➔ Segmentation ➔ Statistics ➔ Recommendations ➔ Display Results.

Dashboard Outputs:
Interactive Flood Mask & Original Overlay
Flood Confidence Percentage
Calculated Flood Area & Coverage
Severity Classification & Contextual Recovery Recommendations
<img width="1895" height="956" alt="Screenshot 2026-07-25 131340" src="https://github.com/user-attachments/assets/4739499f-5ec2-4273-91d1-5efdbc3adf69" />

<img width="1887" height="925" alt="Screenshot 2026-07-25 131349" src="https://github.com/user-attachments/assets/76b3bb3c-b7ad-41c8-830d-8f96211a421b" />


Analysis
Advantages:
Works reliably under cloudy/nighttime conditions.
Provides an automated, end-to-end inference pipeline.
Fast inference speeds (~0.6 to 1 second per image).
User-friendly dashboard with a highly modular architecture.

Limitations:Trained exclusively on the Sen1Floods11 dataset.
Limited to binary flood classification (water vs. no water).
Severity and recovery recommendations rely on simple rule-based thresholds rather than real-time contextual intelligence.
Generalization to other SAR sensors requires further validation.

Future Scope:Implement multi-temporal flood change detection (before vs. after comparisons).
Enable real-time Sentinel-1 data API ingestion.
Expand to multi-class segmentation (identifying water, roads, and buildings).
Integrate with mainstream GIS platforms and mobile applications.
Incorporate LLM-powered, context-aware recovery planning.

Conclusion
This project successfully develops a comprehensive AI-based flood mapping system. By combining a CNN-based flood classifier with Attention U-Net segmentation, the pipeline achieves high segmentation performance with a Mean Dice of 0.8628. Packaged within an interactive Streamlit dashboard, this system effectively demonstrates the feasibility and speed of automated, all-weather flood assessment using Sentinel-1 SAR imagery.
