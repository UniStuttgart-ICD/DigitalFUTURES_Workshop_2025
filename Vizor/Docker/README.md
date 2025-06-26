[FIRST INSTALLATION] 

1. Install Docker Desktop - https://docs.docker.com/desktop/setup/install/windows-install/
2. Include net-tools in your Docker WSL: 
   2a. Open command prompt from your PC
   2b. Run: bash.exe
   2c. Run: sudo apt install net-tools
   2d. Run: exit



[PREREQUISITE] 

1. Make sure Docker Desktop is installed and running. 
2. Connect your PC to the same network as the HoloLens



[INITIAL START]

1. Run command prompt as administrator. 
2. Run: cd <current directory>
3. Run: powershell -ep Bypass .\start_vizor.ps1
4. Keep this window open while using vizor. 

3*. After you have run the powershell once, you can simply run the following for subsequent starts: 
    docker-compose -f vizor_config.yml up



[STOP PROCESS] 

5. Press: ctrl-c
6. Run: docker-compose -f vizor_config.yml down



[CONFIGURE SERVER]

Set the number of devices and other server parameters in "vizor_config.yml". 
**ONLY 4 HOLOLENS CAN BE CONNECTED AT ONE TIME IN THE CURRENT BUILD






Contact Support: xiliu.yang@icd.uni-stuttgart.de