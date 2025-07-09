# ini 파일 설명
## 예제 파일 - classification_mnist.ini
```
[General]
dataset = mnist                          # 사용되는 dataset 이름 (예: MNIST)
class_list = D:/Cpp/TorchSave/Classification/list/mnist.txt  # class 이름이 정리된 텍스트 파일 경로
class_num = 10                           # 총 class 수 (MNIST는 0~9로 10개)
size_h = 28                              # 입력 이미지의 높이
size_w = 28                              # 입력 이미지의 너비
input_channel = 1                        # 입력 이미지 채널 수 (흑백이면 1, RGB이면 3)
gpu_id = 0                               # 사용할 GPU의 ID (0부터 시작. cpu는 -1)
seed_random = false                      # 랜덤 시드 여부 (true면 매 실행마다 랜덤)
seed = 0                                 # 시드 값 (seed_random이 false일 때만 유효, 0 또는 42 권장)

[Training]
train = false                            # 학습 모드 활성화 여부
train_shuffle = true                     # 학습 시 data shuffle 여부
pre_trained = false                      # 사전 학습된 모델을 사용할지 여부
pre_trained_dir = /traced_resnet50.pt    # 사전 학습된 모델의 경로 (pre_trained=true일 때만 사용)
train_dir = ../Classification/datasets/mnist/train          # 학습 데이터가 위치한 dir
epochs = 100                             # 총 학습 epoch 수
batch_size = 32                          # 학습 시 배치 크기
train_load_epoch =                       # 이어서 학습할 epoch (비워두면 새로 학습 시작)
save_epoch = 1                           # 모델 저장 주기 (몇 epoch마다 저장할지)

[Validation]
valid = false                            # 검증 모드 활성화 여부
valid_dir = valid                        # 검증 데이터가 위치한 dir
valid_batch_size = 16                    # 검증 시 배치 크기
valid_freq = 1                           # 검증 수행 주기 (epoch 단위)

[Test]
test = true                              # 테스트 모드 활성화 여부
test_dir = test                          # 테스트 데이터가 위치한 dir
test_load_epoch = latest                 # 테스트에 사용할 epoch (latest면 가장 최근 저장된 모델 사용)
test_result_dir = ../Classification/results  # 테스트 결과 저장 dir

[Network]
lr = 1e-4                                # 학습률 (learning rate)
beta1 = 0.9                              # (고정) Adam 옵티마이저의 beta1 값
beta2 = 0.999                            # (고정) Adam 옵티마이저의 beta2 값
nf = 64                                  # (고정) 네트워크 기본 필터 개수 (예: 첫 Conv layer의 채널 수)
n_layers = 50                            # (고정) 레이어 수 또는 네트워크 깊이 (예: ResNet-50의 경우 50)
```

## 예제 파일 - object_detection_VOC.ini
```
[General]
dataset = object_detection                      # 데이터셋 종류 (여기선 객체 탐지용)
class_list = ../Object_Detection/list/VOC2012.txt  # 클래스 목록이 정의된 파일 경로
anchor_list = ../config/anchor.txt              # 앵커 박스 정의 파일 경로 (YOLO에서 사용)
resize_list = ../config/resize.txt              # 다양한 리사이즈 크기 목록 파일 (멀티스케일 학습 시 사용)
class_num = 20                                  # 클래스 수 (VOC2012 기준 20개 클래스)
size_h = 500                                    # 입력 이미지 리사이즈 높이
size_w = 500                                    # 입력 이미지 리사이즈 너비
input_channel = 3                               # 입력 이미지 채널 수 (흑백이면 1, RGB이면 3)
gpu_id = 0                                      # 사용할 GPU의 ID (0부터 시작. cpu는 -1)
seed_random = false                             # 랜덤 시드 여부 (true면 매 실행마다 랜덤)
seed = 0                                        # 시드 값 (seed_random이 false일 때만 유효, 0 또는 42 권장)
prob_thresh = 0.03                              # (고정) 클래스 확률 임계값 (이 값보다 낮으면 예측 제외)
ignore_thresh = 0.7                             # (고정) 객체 탐지에서 무시할 IOU 임계값 (gt와의 IOU가 작으면 학습에 사용되지 않음)
nms_thresh = 0.5                                # (고정) NMS(Non-Maximum Suppression)에서 사용하는 IOU 임계값
num_anchor = 3                                  # (고정) 한 스케일 당 사용하는 앵커 개수
scales = 3                                      # (고정) 피쳐맵 스케일 개수 (YOLOv3는 3개)

[Training]
train = false                                   # 학습 모드 활성화 여부
train_shuffle = true                            # 학습 시 data shuffle 여부
pre_trained = false                             # 사전 학습된 모델 사용 여부
pre_trained_dir = /yolov3_traced.pt             # 사전 학습된 모델 경로
train_in_dir = trainI                           # 학습용 입력 이미지 디렉토리
train_out_dir = trainO                          # 학습용 라벨(객체 정보) 디렉토리
epochs = 200                                    # 학습 에폭 수
batch_size = 1                                  # 학습 배치 크기
train_load_epoch =                              # 이어서 학습할 에폭 (비워두면 처음부터)
save_epoch = 1                                  # 몇 에폭마다 모델 저장할지
augmentation = true                             # 데이터 증강 사용 여부
# 데이터 증강 관련 파라미터 (확률 기반, 값은 0~1 범위)
jitter = 0.3                                    # (고정) 이미지의 박스 위치를 랜덤으로 변경할 비율
flip_rate = 0.5                                 # (고정) 좌우 반전 확률
scale_rate = 0.5                                # (고정) 스케일 변경 확률
blur_rate = 0.5                                 # (고정) 블러 적용 확률
brightness_rate = 0.5                           # (고정) 밝기 조절 확률
hue_rate = 0.5                                  # (고정) 색조 변화 확률
saturation_rate = 0.5                           # (고정) 채도 변화 확률
shift_rate = 0.5                                # (고정) 이미지 이동 확률
crop_rate = 0.5                                 # (고정) 랜덤 크롭 확률

[Validation]
valid = false                                   # 검증 모드 여부
valid_in_dir = validI                           # 검증용 입력 이미지 디렉토리
valid_out_dir = validO                          # 검증용 라벨 디렉토리
valid_batch_size = 1                            # 검증 시 배치 크기
valid_freq = 1                                  # 검증 수행 주기 (에폭 단위)

[Test]
test = true                                     # 테스트 모드 활성화 여부
test_in_dir = testI                             # 테스트 입력 이미지 디렉토리
test_out_dir = testO                            # 테스트 라벨 디렉토리
test_load_epoch = latest                        # 테스트 시 불러올 에폭 (latest: 가장 최근)
test_result_dir = ../Object_Detection/results   # 테스트 결과 저장 경로

[Detection]
detect = false                                  # 이미지 검출 모드 활성화 여부 (실제 운영 시)
detect_dir = detect/                            # 검출 대상 이미지 디렉토리
detect_load_epoch = 1600                        # 검출 시 사용할 모델 에폭
detect_result_dir = ../Object_Detection/results/detect_result/  # 검출 결과 저장 경로

[Demo]
demo = false                                    # 데모 모드 여부 (웹캠 or 동영상 실시간 검출)
movie =                                         # 검출할 동영상 파일 경로 (비어있으면 웹캠 사용)
demo_load_epoch = 50                            # 데모에서 사용할 모델 에폭
window_w = 1920                                 # (고정) 데모 창 너비
window_h = 1080                                 # (고정) 데모 창 높이
cam_num = 0                                     # (고정) 사용할 카메라 번호

[Network]
lr_init = 1e-4                                  # 초기 학습률
lr_base = 3e-5                                  # 기본 학습률 (warm-up 이후 적용)
lr_decay1 = 1e-5                                # (고정) 학습률 첫 감소 단계
lr_decay2 = 1e-6                                # (고정) 학습률 두 번째 감소 단계
momentum = 0.9                                  # (고정) 옵티마이저 모멘텀 값
weight_decay = 5e-4                             # (고정) 가중치 감소 (정규화 항)
Lambda_coord = 1.0                              # (고정) bbox 좌표 손실 계수
Lambda_object = 1.0                             # (고정) 객체 존재에 대한 손실 계수
Lambda_noobject = 0.1                           # (고정) 객체 없음에 대한 손실 계수
Lambda_class = 1.0                              # (고정) 클래스 예측 손실 계수
```


## 예제 파일 - PASS_Learning_piston.ini
```
[General]
dataset = head				# 사용 중인 데이터셋 이름 (로그 저장 등에 사용)
size_h = 2560				# 입력 이미지 리사이즈 높이 (입력 전 고정 사이즈로 resize)
size_w = 2560				# 입력 이미지 리사이즈 너비
gpu_id = 0			        # 사용할 GPU의 ID (0부터 시작. cpu는 -1)
patch_size = 512			# (권장) 이미지에서 추출할 패치 크기 (GPU Memory 용량에 따라 256, 512, 1024 설정)
seed_random = false			# 랜덤 시드 여부 (true면 매 실행마다 랜덤)
seed = 0				# 시드 값 (seed_random이 false일 때만 유효, 0 또는 42 권장)

[Training]
train = true				# 학습 실행 여부 (true, false)
mode = unsuper				# 학습 모드 (unsuper: 비지도 학습, super: 지도 학습)
epochs = 50				# 학습 epoch 수
pretrain_mode = unsuper		        # pretrain model의 mode (unsuper, super 교차 가능)
train_load_epoch = latest	        # 이어서 학습할 epoch (비워두면 처음부터 학습, latest or 숫자)
train_super_dir = ..\Data\piston_image\head\notfound\origin	# 지도 학습 이미지 dir (정상/불량 포함)
train_unsuper_dir = ..\Data\piston_image\head\overkill\origin	# 비지도 학습 이미지 dir (보통 정상 이미지만 사용)
batch_size = 1				# 학습 배치 크기
save_epoch = 1				# 모델 저장 주기
unsuper_count = 100			# 비지도 모델 저장 주기(data-level)
l1_weight=0.6				# (권장) L1 Loss 가중치
focal_weight=0.4			# (권장) Focal Loss 가중치

[Validation]
valid = false				# 검증 실행 여부 (true, false)
valid_dir = ..\Data\piston_image\head\notfound\origin	# 검증용 이미지 dir
valid_batch_size = 1		        # (고정) 검증 배치 크기
valid_freq = 1				# (고정) 모델 검증 주기

[Test]
test = true				# 테스트 실행 여부 (true, false)
test_dir = ..\Data\piston_image\Data\head\test__\origin		# 테스트용 이미지 디렉토리
test_load_epoch = latest	        # 테스트 시 사용할 모델 epoch (latest or 숫자)
test_result_dir = ../PASS_Learning/result	# 테스트 결과 저장 dir

threshold = 0.1				# (권장) 이상 판단 임계값 (0~1 사이, score threshold)

[Network]
lr = 5e-5			        # 최대 learning rate (권장 값: 3e-5 ~ 1e-4)
min_lr = 5E-7				# 최소 learning rate (scheduler 사용 시) (권장 값: 5e-7 ~ 1e-6)
warmup_ratio = 0			# warmup 비율 (0이면 warmup 없음)
beta1 = 0.5				# (고정) Adam optimizer 파라미터 beta1
beta2 = 0.999				# (고정) Adam optimizer 파라미터 beta2
```

## 예제 파일 - segmentation.ini
```
[General]
dataset = segmentation                     # 데이터셋 타입 (세그멘테이션 작업용)
size_h = 256                               # 입력 이미지 리사이즈 높이
size_w = 256                               # 입력 이미지 리사이즈 너비
input_channel = 3                          # 입력 이미지 채널 수  (흑백이면 1, RGB이면 3)
latent_space = 512                         # 잠재 공간 차원 수 (디코더 병목 벡터 크기)
class_num = 3                              # 분할 대상 클래스 수 (배경 포함)
gpu_id = 0                                 # 사용할 GPU의 ID (0부터 시작. cpu는 -1)
seed_random = false                        # 랜덤 시드 여부 (true면 매 실행마다 랜덤)
seed = 0                                   # 시드 값 (seed_random이 false일 때만 유효, 0 또는 42 권장)

[Training]
train = true                               # 학습 모드 활성화 여부
train_in_dir = trainI                      # 학습용 입력 이미지 디렉토리
train_out_dir = trainO                     # 학습용 라벨(세그멘테이션 마스크) 디렉토리
epochs = 300                               # 학습 에폭 수
batch_size = 2                             # 학습 시 배치 크기
train_load_epoch = latest                  # 이어서 학습할 에폭 번호 (비워두면 처음부터 학습)
save_epoch = 5                             # 몇 에폭마다 모델을 저장할지 설정

[Validation]
valid = false                              # 검증 모드 활성화 여부
valid_in_dir = validI                      # 검증용 입력 이미지 디렉토리
valid_out_dir = validO                     # 검증용 라벨 디렉토리
valid_batch_size = 1                       # 검증 시 배치 크기
valid_freq = 1                             # 검증 수행 주기 (에폭 단위)

[Test]
test = true                                # 테스트 모드 활성화 여부
test_in_dir = testI                        # 테스트 입력 이미지 디렉토리
test_out_dir = testO                       # 테스트 라벨 디렉토리
test_load_epoch = latest                   # 테스트에 사용할 모델 에폭 (latest: 가장 최근 모델)
test_result_dir = ../Segmentation/results  # 테스트 결과 저장 디렉토리

[Network]
lr = 1e-4                                  # learning rate (권장 값: 3e-5 ~ 1e-4)
no_dropout = false                         # 드롭아웃 사용 여부 (true면 사용 안 함)
beta1 = 0.5                                # (고정) Adam 옵티마이저 beta1 값
beta2 = 0.999                              # (고정) Adam 옵티마이저 beta2 값
nf = 64                                    # (고정) 네트워크 기본 필터 수 (예: U-Net 첫 Conv 채널 수)
```