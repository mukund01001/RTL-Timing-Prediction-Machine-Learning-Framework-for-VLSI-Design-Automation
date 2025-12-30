# ML Framework for RTL Timing Prediction

## Overview
Machine learning framework for predicting RTL circuit timing with SHAP/LIME explainability.

## Objectives
- Predict timing slack/delay with <10% error ✅ **ACHIEVED**
- Global interpretability using SHAP ✅ **COMPLETED**
- Local interpretability using LIME ✅ **COMPLETED**

## Results
- **MAPE:** <6% (exceeds target)
- **R² Score:** >0.92
- **Dataset:** 20 diverse RTL designs
- **Models:** Random Forest, Gradient Boosting

## Files
- `Problem6_RTL_Timing_Prediction.ipynb` - Complete analysis
- `datasets/` - RTL timing data
- `models/` - Trained ML models
- `visualizations/` - SHAP/LIME plots
- `rtl_designs/` - Source Verilog files

## How to Run
1. Open notebook in Google Colab or Jupyter
2. Upload datasets to Google Drive (or adjust file paths)
3. Run all cells sequentially
4. View results and visualizations

## Technologies
- **HDL:** Verilog
- **Synthesis:** Xilinx Vivado
- **ML:** scikit-learn, XGBoost
- **Explainability:** SHAP, LIME
- **Visualization:** Matplotlib, Seaborn

