@echo off
echo ============================================
echo Deploying TFT Training to AWS Instance
echo ============================================
echo.

set KEY=C:\Users\babas\Downloads\cortafy-key.pem
set HOST=ubuntu@98.87.133.69
set SCP=scp -i "%KEY%" -o StrictHostKeyChecking=no
set SSH=ssh -i "%KEY%" -o StrictHostKeyChecking=no %HOST%

echo [1/5] Creating remote directory...
%SSH% "mkdir -p ~/tft_training/results"

echo [2/5] Copying data file (~15MB)...
%SCP% "c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\patent1-energy-orchestration\data\merged_enriched_2020_2025.csv" %HOST%:~/tft_training/

echo [3/5] Copying model code...
%SCP% "c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\patent1-energy-orchestration\src\tft_model.py" %HOST%:~/tft_training/
%SCP% "c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\patent1-energy-orchestration\src\train_tft.py" %HOST%:~/tft_training/

echo [4/5] Installing dependencies and starting training...
%SSH% "cd ~/tft_training && pip3 install torch numpy pandas scikit-learn --quiet 2>/dev/null; nohup python3 train_tft_standalone.py > training_log.txt 2>&1 &"

echo [5/5] Done! Training running in background.
echo.
echo To check progress: ssh -i "%KEY%" %HOST% "tail -20 ~/tft_training/training_log.txt"
echo To get results: scp -i "%KEY%" %HOST%:~/tft_training/results/* "c:\Users\babas\Dev_Projects\Optena Data Center Energy Optimization\patent1-energy-orchestration\results\"
echo.
pause
