# Classification, Object Detection, Segmentation, Anomaly Detection에 대한 C++ 코드 알고리즘

## 실행에 필요한 개발환경
- python, cuda, cudnn, libtorch, opencv, boost 등
- 설치 버전 : python 3.11, cuda 12.1, pip install (torch 2.1.2, transformer 4.51.3, diffusers 0.33.1) → 버전 미일치 (Error)

## 시스템 환경 변수 설정 (Path)
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64
- C:\Program Files\Python311\Scripts\
- C:\Program Files\Python311\DLLS\
- C:\Program Files\Python311\
- C:\Users\JHCHUN\AppData\Roaming\Python\Python311\Scripts
- C:\Users\JHCHUN\AppData\Roaming\Python\Python311\site-packages\torch\lib

## 실행 방법
### 1. 각 알고리즘에 대한 dll 생성
#### 1.1. 종류
- Classification
- Object Detection
- Segmentation
- Anomaly Detection (PASS Learning)

#### 1.2. dll 생성
- Visual Studio를 열어 개발 환경을 세팅한 후 dll 생성
- dll 생성 path는 상위 폴더로 설정 (모든 dll 한 폴더에 모이도록)

### 2. Model 실행
#### 2.1. Visual Studio를 열어 개발 환경을 세팅한 후 dll 생성
1. 🔨 기본 개발환경 세팅
Visual Studio 버전: Visual Studio 2019 이상

플랫폼 도구 집합: v142 (또는 설치된 최신 버전)
플랫폼: x64

구성(Configuration): Release or Debug (학습 시엔 Release 권장)

2. 📦 LibTorch 세팅
   
2.1. 프로젝트 속성 설정
▸ 속성 → VC++ 디렉터리

C/C++ → 포함 디렉터리 (Include Directories):
```
C:\Program Files\Python311\include
C:\Users\JHCHUN\AppData\Roaming\Python\Python311\site-packages\torch\include
C:\Users\JHCHUN\AppData\Roaming\Python\Python311\site-packages\torch\csrc\api\include
▼ TorchSave Only
D:\Cpp\TorchSave\Classification
D:\Cpp\TorchSave\Object_Detection
D:\Cpp\TorchSave\Segmentation
D:\Cpp\TorchSave\PASS_Learning
```

C/C++ → 전처리기:
```
(추가) _CRT_SECURE_NO_WARNINGS;
```
포함 디렉터리 (Include Directories):
```
$(ProjectDir)libs\libtorch\include
$(ProjectDir)libs\libtorch\include\torch\csrc\api\include
```

라이브러리 디렉터리 (Library Directories):
```
C:\Program Files\Python311\libs
C:\Users\JHCHUN\AppData\Roaming\Python\Python311\site-packages\torch\lib
▼ TorchSave Only
D:\Cpp\TorchSave\x64\Release
```
▸ 속성 → 링커 → 입력

추가 종속성:
```
python311.lib
torch.lib
torch_cpu.lib
c10.lib
(cuda 사용시 아래 추가)
torch_cuda.lib
c10_cuda.lib
▼ TorchSave Only
Classification.lib
Object_Detection.lib
Segmentation.lib
PASS_Learning.lib
```

▸ 속성 → C/C++ → 코드 생성
런타임 라이브러리: Multi-threaded (/MT) 또는 /MD (libtorch 빌드에 맞게)

▸ 속성 → 링커 → 시스템
하위 시스템: 콘솔(/SUBSYSTEM:CONSOLE)

▸ 속성 → 링커 → 고급
입력 파일 자동 복사 설정 (DLL 문제 방지):
libs/libtorch/bin/*.dll 파일을 .exe와 같은 위치로 복사

#### 2.2. config/ini 파일을 통해 train/test 설정하여 동작
- ini 파일에 train 또는 test의 true/false를 사용하여 모드 변경

#### 2.3. 세부적인 조정값들은 ini 파일로 변경 가능
- dataset의 경우 이미지 파일이 있는 상위 폴더를 지정
- supervised dataset에서 notfound의 경우 origin과 label 폴더를 같은 경로에 설정
- 추가적인 세부 정보들은 [ini 설명 파일](./config/README.md) 참고 

#### 2.4. TorchSave 폴더 내에 있는 코드를 Run 하면 각 알고리즘에 대해 모델이 동작