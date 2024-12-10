# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk, messagebox
import random
import json
import os

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우
# plt.rcParams['font.family'] = 'AppleGothic'  # macOS의 경우
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 신경망 클래스 정의
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 가중치 초기화
        self.weights = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, hidden_size) * 0.1)
            prev_size = hidden_size
        self.weights.append(np.random.randn(prev_size, output_size) * 0.1)
        
        # 각 층의 활성화값을 저장할 변수
        self.activations = [None] * (len(hidden_sizes) + 2)
        
        # 학습 히스토리 저장
        self.weight_history = []
        self.error_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # 입력층
        self.activations[0] = np.array(x)
        
        # 은닉층들과 출력층
        for i in range(len(self.weights)):
            self.activations[i+1] = self.sigmoid(
                np.dot(self.activations[i], self.weights[i])
            )
        
        return self.activations[-1]

    def train(self, inputs, targets):
        batch_size = inputs.shape[0]
        
        # 현재 에포크에 따른 학습률 조정 (더 안정적인 감소)
        current_lr = self.learning_rate * (0.99 ** (len(self.error_history) / 2000))
        current_lr = max(0.001, current_lr)  # 최소 학습률 설정
        
        # Forward pass
        layer_outputs = [[] for _ in range(len(self.weights) + 1)]
        layer_outputs[0] = inputs
        
        current_input = inputs
        for i, weight in enumerate(self.weights):
            current_output = self.sigmoid(np.dot(current_input, weight))
            layer_outputs[i + 1] = current_output
            current_input = current_output
        
        outputs = layer_outputs[-1]
        
        # 역전파
        output_errors = targets - outputs
        deltas = [output_errors * self.sigmoid_derivative(outputs)]
        
        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(deltas[0], self.weights[i].T)
            delta = error * self.sigmoid_derivative(layer_outputs[i])
            deltas.insert(0, delta)
        
        # 가중치 업데이트 (그래디언트 클리핑 추가)
        for i in range(len(self.weights)):
            layer_input = layer_outputs[i]
            layer_delta = deltas[i]
            gradient = np.dot(layer_input.T, layer_delta)
            # 그래디언트 클리핑
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 1.0:
                gradient = gradient / gradient_norm
            self.weights[i] += current_lr * gradient
        
        # 학습 상태 저장
        self.weight_history.append([w.copy() for w in self.weights])
        mean_error = np.mean(np.abs(output_errors))
        self.error_history.append(mean_error)
        
        return mean_error

    def get_network_state(self):
        return {
            'activations': self.activations,
            'weights': self.weights
        }

    def reset_weights(self):
        """가중치 초기화"""
        self.weights = []
        prev_size = self.input_size
        for hidden_size in self.hidden_sizes:
            self.weights.append(np.random.randn(prev_size, hidden_size) * 0.1)
            prev_size = hidden_size
        self.weights.append(np.random.randn(prev_size, self.output_size) * 0.1)
        self.activations = [None] * (len(self.hidden_sizes) + 2)
        self.weight_history = []  # 가중치 히스토리 초기화 추가
        self.error_history = []   # 에러 히스토리 초기화 추가

# GUI 클래스 정의
class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("신경망 학습 시각화 시스템")
        
        # 신경망 초기화 (더 큰 구조)
        self.nn = SimpleNeuralNetwork(
            input_size=6,
            hidden_sizes=[12, 8, 6],
            output_size=3,
            learning_rate=0.05
        )
        
        # 학습 데이터 저장
        self.batch_size = 2  # 배치 크기 추가
        self.training_data = []
        self.current_data_index = 0
        self.epoch_count = 0
        self.error_history = []
        self.total_epochs = 1000  # 전체 학습 에포크 수 설정
        
        # 미리 정의된 데이터셋
        self.predefined_datasets = {
            # 기본 패턴 인식
            "기본 패턴": {
                "inputs": [
                    [1, 1, 1, 0, 0, 0],  # 왼쪽 활성화
                    [0, 0, 0, 1, 1, 1],  # 오른쪽 활성화
                    [1, 0, 1, 0, 1, 0],  # 교차 패턴
                    [0, 1, 0, 1, 0, 1],  # 반대 교차
                    [1, 1, 0, 0, 1, 1],  # 양끝 패성화
                    [0, 0, 1, 1, 0, 0],  # 중앙 활성화
                ],
                "targets": [
                    [1, 0, 0],  # 왼쪽 패턴
                    [0, 1, 0],  # 오른쪽 패턴
                    [0, 0, 1],  # 교차 패턴
                    [1, 1, 0],  # 혼합 패턴 1
                    [0, 1, 1],  # 혼합 패턴 2
                    [1, 0, 1],  # 혼합 패턴 3
                ]
            },
            
            # 대칭성 인식
            "대칭 패턴": {
                "inputs": [
                    [1, 0, 0, 0, 0, 1],  # 완전 대칭
                    [0, 1, 0, 0, 1, 0],  # 내부 대칭
                    [0, 0, 1, 1, 0, 0],  # 중앙 대칭
                    [1, 1, 0, 0, 1, 1],  # 대칭 + 강한 신호
                    [0, 1, 1, 1, 1, 0],  # 대칭 + 중앙 강조
                    [1, 0, 1, 1, 0, 1],  # 대칭 + 교차
                ],
                "targets": [
                    [1, 1, 1],  # 완벽한 대칭
                    [1, 0, 1],  # 부분 대칭
                    [0, 1, 1],  # 중앙 중심 대칭
                    [1, 1, 0],  # 강한 대칭
                    [0, 0, 1],  # 약한 대칭
                    [1, 0, 0],  # 불완전 대칭
                ]
            },
            
            # 순차 패턴
            "순차 패턴": {
                "inputs": [
                    [1, 1, 0, 0, 0, 0],  # 왼쪽 시작
                    [0, 1, 1, 0, 0, 0],  # 왼쪽에서 이동
                    [0, 0, 1, 1, 0, 0],  # 중앙
                    [0, 0, 0, 1, 1, 0],  # 오른쪽으로 이동
                    [0, 0, 0, 0, 1, 1],  # 오른쪽 끝
                    [1, 0, 0, 0, 0, 1],  # 양끝
                    [0, 1, 1, 1, 1, 0],  # 중앙 집중
                    [1, 1, 1, 0, 0, 0],  # 왼쪽 집중
                ],
                "targets": [
                    [1, 0, 0],  # 왼쪽 영역
                    [1, 1, 0],  # 왼쪽-중앙
                    [0, 1, 0],  # 중앙 영역
                    [0, 1, 1],  # 중앙-오른쪽
                    [0, 0, 1],  # 오른쪽 영역
                    [1, 0, 1],  # 양끝 영역
                    [0, 1, 0],  # 중앙 강조
                    [1, 0, 0],  # 왼쪽 강조
                ]
            },
            
            # 강도 패턴
            "강도 패턴": {
                "inputs": [
                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],  # 약한 신호
                    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # 중간 신호
                    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],  # 강한 신호
                    [0.2, 0.4, 0.6, 0.6, 0.4, 0.2],  # 중앙 강조
                    [0.8, 0.6, 0.4, 0.4, 0.6, 0.8],  # 외곽 강조
                    [0.2, 0.8, 0.2, 0.8, 0.2, 0.8],  # 교차 강도
                ],
                "targets": [
                    [0.2, 0.2, 0.2],  # 약한 반응
                    [0.5, 0.5, 0.5],  # 중간 반응
                    [0.8, 0.8, 0.8],  # 강한 반응
                    [0.3, 0.8, 0.3],  # 중앙 중심
                    [0.8, 0.3, 0.8],  # 외곽 중심
                    [0.5, 0.5, 0.8],  # 복합 반응
                ]
            }
        }
        
        # GUI 레이아웃 설정
        self.setup_gui()
        self.setup_error_plot()
        
        # 애니메이션 시작
        self.is_training = False
        self.frame_count = 0
        self.ani = None
        
        # 윈도우 종료 이벤트 처리
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.start_animation()

    def setup_gui(self):
        # 메인 컨테이너
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 좌측 패널 (컨트롤)
        left_panel = ttk.LabelFrame(main_container, text="제어 패널")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 입력 섹션
        input_frame = ttk.LabelFrame(left_panel, text="입력값 설정")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_vars = []
        for i in range(6):
            frame = ttk.Frame(input_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"입력 {i+1}:").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            self.input_vars.append(var)
            ttk.Scale(frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                     variable=var, command=self.update_network).pack(
                         side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Label(frame, textvariable=var, width=5).pack(side=tk.RIGHT)
        
        # 목표값 섹션
        target_frame = ttk.LabelFrame(left_panel, text="목표값 설정")
        target_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.target_vars = []
        for i in range(3):
            frame = ttk.Frame(target_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"목표 {i+1}:").pack(side=tk.LEFT)
            var = tk.DoubleVar(value=0.0)
            self.target_vars.append(var)
            ttk.Scale(frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                     variable=var, command=self.update_network).pack(
                         side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Label(frame, textvariable=var, width=5).pack(side=tk.RIGHT)
        
        # 학습 제어 섹션
        control_frame = ttk.LabelFrame(left_panel, text="학습 제어")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 학습률 설정
        lr_frame = ttk.Frame(control_frame)
        lr_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(lr_frame, text="학습률:").pack(side=tk.LEFT)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Scale(lr_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL,
                 variable=self.lr_var, command=self.update_learning_rate).pack(
                     side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(lr_frame, textvariable=self.lr_var, width=5).pack(side=tk.RIGHT)
        
        # 학습 모드 선택
        mode_frame = ttk.LabelFrame(control_frame, text="학습 모드")
        mode_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.train_mode = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="단일 입력 학습", 
                       variable=self.train_mode, value="single").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="전체 데이터 학습", 
                       variable=self.train_mode, value="all").pack(anchor=tk.W)
        
        # 배치 크기 설정
        batch_frame = ttk.Frame(mode_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(batch_frame, text="배치 크기:").pack(side=tk.LEFT)
        self.batch_var = tk.IntVar(value=2)
        ttk.Spinbox(batch_frame, from_=1, to=10, 
                    textvariable=self.batch_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(batch_frame, text="(1~전체 데이터 크기)").pack(side=tk.LEFT)
        
        # 전체 에포크 설정
        epoch_frame = ttk.Frame(mode_frame)
        epoch_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(epoch_frame, text="전체 에포크:").pack(side=tk.LEFT)
        self.total_epochs_var = tk.IntVar(value=1000)
        ttk.Entry(epoch_frame, textvariable=self.total_epochs_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # 데이터 관리 버튼
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        self.train_button = ttk.Button(button_frame, text="학습 시작", 
                                     command=self.toggle_training)
        self.train_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="데이터 저장", 
                  command=self.save_current_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="초기화", 
                  command=self.clear_training_data).pack(side=tk.LEFT, padx=2)
        
        # 학습 상태 표시
        status_frame = ttk.LabelFrame(left_panel, text="학습 상태")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_text = tk.StringVar(value="준비")
        ttk.Label(status_frame, textvariable=self.status_text).pack(pady=5)
        
        self.epoch_text = tk.StringVar(value="에포크: 0")
        ttk.Label(status_frame, textvariable=self.epoch_text).pack(pady=5)
        
        self.error_text = tk.StringVar(value="오차: 0.0")
        ttk.Label(status_frame, textvariable=self.error_text).pack(pady=5)
        
        # 우측 패널 (시각화)
        right_panel = ttk.LabelFrame(main_container, text="신경망 구조")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # matplotlib 그림 초기화
        self.fig = plt.Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        dataset_frame = ttk.LabelFrame(left_panel, text="데이터셋 선택")
        dataset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 데이터셋 콤보박스
        self.dataset_var = tk.StringVar()
        dataset_combo = ttk.Combobox(dataset_frame, 
                                textvariable=self.dataset_var,
                                values=list(self.predefined_datasets.keys()))
        dataset_combo.pack(fill=tk.X, padx=5, pady=5)
        dataset_combo.bind('<<ComboboxSelected>>', self.load_dataset)
        
        # 데이터셋 정보 표시
        self.dataset_info = tk.StringVar(value="데이터 없음")
        ttk.Label(dataset_frame, textvariable=self.dataset_info).pack(pady=5)
        
        # 데이터셋 로드/저장 버튼
        button_frame = ttk.Frame(dataset_frame)
        button_frame.pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="데이터셋 로드",
                  command=self.load_selected_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="현재 데이터 추가",
                  command=self.save_to_dataset).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="데이터 보기",
                  command=self.show_dataset_contents).pack(side=tk.LEFT, padx=2)
        
    


    def setup_error_plot(self):
        """오차 그래프 설정"""
        self.error_fig = plt.Figure(figsize=(10, 3))
        self.error_ax = self.error_fig.add_subplot(111)
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, master=self.root)
        self.error_canvas.get_tk_widget().pack(fill=tk.X, expand=False, padx=10, pady=5)

    def update_network(self, *args):
        # 슬라이더 값으로 입력 업데이트
        inputs = np.array([[var.get() for var in self.input_vars]])
        self.nn.forward(inputs)
        self.draw_network()

    def update_learning_rate(self, *args):
        try:
            new_lr = float(self.lr_var.get())
            if 0 < new_lr <= 1.0:
                self.nn.learning_rate = new_lr
                self.status_text.set(f"학습률이 {new_lr:.4f}로 변경됨")
            else:
                self.status_text.set("학습률은 0과 1 사이여야 합니다")
        except ValueError:
            self.status_text.set("올바른 학습률 값을 입력하세요")

    def toggle_training(self):
        self.is_training = not self.is_training
        self.train_button.config(text="학습 중지" if self.is_training else "학습 시작")

    def draw_network(self):
        self.ax.clear()
        
        # 신경망 상태 가져오기
        state = self.nn.get_network_state()
        
        # 노드 위치 계산 (각 층별)
        layer_positions = []
        max_nodes = max([self.nn.input_size] + self.nn.hidden_sizes + [self.nn.output_size])
        
        # 입력층 위치
        layer_positions.append(np.array([[0, i*1.5] for i in range(self.nn.input_size)]))
        
        # 은닉층들의 위치
        for i, hidden_size in enumerate(self.nn.hidden_sizes, 1):
            layer_positions.append(np.array([[i, j*1.5] for j in range(hidden_size)]))
        
        # 출력층 위치
        layer_positions.append(np.array([[len(self.nn.hidden_sizes)+1, i*1.5] 
                                       for i in range(self.nn.output_size)]))
        
        # 가중치 선 그리기
        for layer in range(len(state['weights'])):
            for i in range(len(layer_positions[layer])):
                for j in range(len(layer_positions[layer+1])):
                    weight = state['weights'][layer][i, j]
                    color = 'red' if weight < 0 else 'blue'
                    alpha = min(abs(weight), 1.0)
                    self.ax.plot([layer_positions[layer][i,0], layer_positions[layer+1][j,0]],
                               [layer_positions[layer][i,1], layer_positions[layer+1][j,1]],
                               color=color, alpha=alpha, linewidth=1)
        
        # 노드 그리기 및 값 표시
        colors = ['green'] + ['blue']*(len(self.nn.hidden_sizes)) + ['red']
        labels = ['입력층'] + [f'은닉층 {i+1}' for i in range(len(self.nn.hidden_sizes))] + ['출력층']
        
        # 현재 입력값으로 순전파 수행하여 활성화 값 업데이트
        if not self.is_training:  # 학습 중이 아닐 때만 현재 입력값으로 업데이트
            inputs = np.array([[var.get() for var in self.input_vars]])
            self.nn.forward(inputs)
        
        # 노드 그리기 및 활성화 값 표시
        for layer, (pos, color, label) in enumerate(zip(layer_positions, colors, labels)):
            self.ax.scatter(pos[:,0], pos[:,1], c=color, s=150, label=label)
            if state['activations'][layer] is not None:
                activations = state['activations'][layer]
                if len(activations.shape) == 1:
                    activations = activations.reshape(1, -1)
                for i, node_pos in enumerate(pos):
                    value = activations[0, i]
                    self.ax.annotate(f'{value:.2f}',
                                   (node_pos[0], node_pos[1]),
                                   xytext=(10, 5),
                                   textcoords='offset points')
        
        # 그래프 설정
        self.ax.set_title('신경망 구조')
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.ax.set_xlim(-0.5, len(self.nn.hidden_sizes)+1.5)
        self.ax.set_ylim(-1, max_nodes*1.5)
        
        self.canvas.draw()

    def animate(self, frame):
        if self.is_training:
            for _ in range(10):  # 한 프레임당 10번의 학습 수행
                self.epoch_count += 1
                self.epoch_text.set(f"에포크: {self.epoch_count}")
                
                error = 0
                if self.train_mode.get() == "single":
                    # 단일 입력 학습
                    inputs = np.array([[var.get() for var in self.input_vars]])
                    targets = np.array([[var.get() for var in self.target_vars]])
                    error = self.nn.train(inputs, targets)
                    self.nn.forward(inputs)  # 시각화 업데이트
                else:
                    # 전체 데이터 배치 학습
                    if len(self.training_data) > 0:
                        try:
                            # 배치 크기 조정 (최소 2, 최대 데이터 크기의 1/2)
                            max_batch = max(2, len(self.training_data) // 2)
                            batch_size = min(max_batch, self.batch_var.get())
                            
                            # 배치 데이터 준비
                            indices = np.random.choice(len(self.training_data), batch_size, replace=False)
                            batch_inputs = []
                            batch_targets = []
                            
                            for idx in indices:
                                inputs, targets = self.training_data[idx]
                                batch_inputs.append(inputs)
                                batch_targets.append(targets)
                            
                            # 배치 학습
                            inputs = np.array(batch_inputs, dtype=np.float32)
                            targets = np.array(batch_targets, dtype=np.float32)
                            error = self.nn.train(inputs, targets)
                            
                            # 전체 데이터에 대한 오차 계산
                            all_inputs = np.array([data[0] for data in self.training_data])
                            all_targets = np.array([data[1] for data in self.training_data])
                            outputs = self.nn.forward(all_inputs)
                            error = np.mean(np.abs(all_targets - outputs))
                            
                            # 시각화 업데이트 (첫 번째 데이터로)
                            self.nn.forward(inputs[0:1])
                            
                        except Exception as e:
                            print(f"학습 중 오류 발생: {e}")
                            self.is_training = False
                            self.train_button.config(text="학습 시작")
                            self.status_text.set("학습 오류 발생")
                            break
                
                self.error_history.append(error)
                self.error_text.set(f"오차: {error:.6f}")
                self.update_error_plot()
                
                # 학습 상태 업데이트
                progress = (self.epoch_count / self.total_epochs_var.get()) * 100
                self.status_text.set(f"학습 중... ({progress:.1f}%)")
                
                # 학습 종료 조건 체크
                if self.epoch_count >= self.total_epochs_var.get() or error < 0.001:  # 오차가 충분히 작으면 종료
                    self.is_training = False
                    self.train_button.config(text="학습 시작")
                    self.status_text.set("학습 완료")
                    break
                
                # 주기적으로 네트워크 시각화 업데이트
                if self.epoch_count % 5 == 0:
                    self.draw_network()

    def start_animation(self):
        """애니메이션 시작"""
        if self.ani is not None:
            self.ani.event_source.stop()
            self.ani = None
        
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.animate, 
            interval=50,
            blit=False,
            cache_frame_data=False  # 캐시 비활성화
        )
        
        # 캔버스 업데이트
        self.canvas.draw()

    def update_error_plot(self):
        """오차 그래프 업데이트"""
        try:
            self.error_ax.clear()
            if self.error_history:
                self.error_ax.plot(self.error_history[-100:], 'b-', linewidth=1)
                self.error_ax.set_title('학습 오차')
                self.error_ax.set_xlabel('에포크')
                self.error_ax.set_ylabel('오차')
                self.error_ax.grid(True)
                
                # y축 범위 설정 개선
                if len(self.error_history) > 1:
                    min_error = min(self.error_history[-100:])
                    max_error = max(self.error_history[-100:])
                    
                    # min과 max가 같을 때 처리
                    if min_error == max_error:
                        if min_error == 0:
                            self.error_ax.set_ylim([-0.1, 0.1])
                        else:
                            # 현재 값의 ±10% 범위 설정
                            margin = abs(min_error) * 0.1
                            self.error_ax.set_ylim([min_error - margin, max_error + margin])
                    else:
                        margin = (max_error - min_error) * 0.1
                        self.error_ax.set_ylim([min_error - margin, max_error + margin])
                
                self.error_canvas.draw()
        except Exception as e:
            print(f"오차 그래프 업데이트 중 오류 발생: {e}")
    
    def on_closing(self):
        if self.ani is not None:
            self.ani.event_source.stop()
        self.root.destroy()

    def save_current_data(self):
        """현재 입력값과 목표값을 학습 데이터로 저장"""
        inputs = [var.get() for var in self.input_vars]
        targets = [var.get() for var in self.target_vars]
        self.training_data.append((inputs, targets))
        self.status_text.set(f"데이터 {len(self.training_data)}개 저장됨")
        print(f"데이터 포인트 {len(self.training_data)}개 저장됨")

    def load_dataset(self, event=None):
        """선택된 데이터셋 정보 표시"""
        selected = self.dataset_var.get()
        if selected in self.predefined_datasets:
            dataset = self.predefined_datasets[selected]
            info = f"데이터셋: {selected}\n"
            info += f"데이터 개수: {len(dataset['inputs'])}\n"
            info += f"입력 크기: {len(dataset['inputs'][0])}\n"
            info += f"출력 크기: {len(dataset['targets'][0])}"
            self.dataset_info.set(info)

    def load_selected_dataset(self):
        """선택된 데이터셋을 학습 데이터로 로드"""
        try:
            selected = self.dataset_var.get()
            if not selected:
                self.status_text.set("데이터셋을 선택해주세요")
                return
            if selected in self.predefined_datasets:
                dataset = self.predefined_datasets[selected]
                
                # 기존 학습 상태 초기화 여부 확인
                if self.epoch_count > 0:
                    if messagebox.askyesno("초기화 확인", 
                        "현재 학습 상태를 초기화하시겠습니까?\n'아니오'를 선택하면 현재 가중치를 유지합니다."):
                        self.nn.reset_weights()
                        self.error_history = []
                        self.nn.error_history = []
                
                # 데이터 로드
                self.training_data = list(zip(dataset['inputs'], dataset['targets']))
                self.epoch_count = 0
                self.current_data_index = 0
                self.epoch_text.set("에포크: 0")
                self.status_text.set(f"{selected} 데이터셋 로드됨 ({len(self.training_data)}개)")
                
                # 첫 번째 데이터 표시
                if self.training_data:
                    inputs, targets = self.training_data[0]
                    for i, value in enumerate(inputs):
                        if i < len(self.input_vars):
                            self.input_vars[i].set(value)
                    for i, value in enumerate(targets):
                        if i < len(self.target_vars):
                            self.target_vars[i].set(value)
                        
                # 시각화 업데이트
                self.draw_network()
                self.update_error_plot()
                
        except Exception as e:
            self.status_text.set(f"데이터셋 로드 오류: {str(e)}")
            print(f"데이터셋 로드 중 오류 발생: {e}")

    def save_to_dataset(self):
        """현재 입력/목표값을 새로운 데이터셋으로 저장"""
        selected = self.dataset_var.get()
        if not selected:
            selected = f"사용자정의_{len(self.predefined_datasets)+1}"
        
        inputs = [var.get() for var in self.input_vars]
        targets = [var.get() for var in self.target_vars]
        
        if selected not in self.predefined_datasets:
            self.predefined_datasets[selected] = {
                "inputs": [inputs],
                "targets": [targets]
            }
        else:
            self.predefined_datasets[selected]["inputs"].append(inputs)
            self.predefined_datasets[selected]["targets"].append(targets)
        
        # 콤보박스 업데이트
        dataset_values = list(self.predefined_datasets.keys())
        self.dataset_var.set(selected)
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Combobox):
                child['values'] = dataset_values
        
        self.status_text.set(f"데이터가 {selected} 데이터셋에 추가됨")

    def clear_training_data(self):
        """저장된 학습 데이터 초기화"""
        # 애니메이션 중지
        if self.ani is not None:
            self.ani.event_source.stop()
            self.ani = None
        
        if self.training_data:  # 학습 데이터가 있는 경우
            # 사용자에게 확인
            if tk.messagebox.askyesno("초기화 확인", 
                "학습 데이터도 함께 초기화하시겠습니까?\n'아니오'를 선택하면 에포크만 초기화됩니다."):
                # 전체 초기화
                self.training_data = []
                self.nn.reset_weights()
                self.status_text.set("데이터와 가중치가 초기화됨")
            else:
                # 에포크만 초기화
                self.status_text.set("에포크가 초기화됨")
        
        # 공통 초기화 항목
        self.current_data_index = 0
        self.epoch_count = 0
        self.error_history = []
        self.nn.error_history = []
        self.nn.weight_history = []  # 가중치 히스토리 초기화 추가
        self.epoch_text.set("에포크: 0")
        self.error_text.set("오차: 0.0")
        self.is_training = False  # 학습 상태 초기화
        self.train_button.config(text="학습 시작")  # 버튼 텍스트 초기화
        
        # 시각화 업데이트
        self.draw_network()
        self.update_error_plot()
        
        # 애니메이션 재시작
        self.start_animation()

    def show_dataset_contents(self):
        """데이터셋 내용을 보여주는 새 창 생성"""
        selected = self.dataset_var.get()
        if not selected:
            self.status_text.set("데이터셋을 선택해주세요")
            return
        
        if selected not in self.predefined_datasets:
            self.status_text.set("선택된 데이터셋이 없습니다")
            return
        
        # 새 창 생성
        data_window = tk.Toplevel(self.root)
        data_window.title(f"데이터셋 내용 - {selected}")
        data_window.geometry("600x400")
        
        # 스크롤바가 있는 텍스트 영역 생성
        frame = ttk.Frame(data_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(frame, wrap=tk.NONE)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 스크롤바 추가
        scrollbar_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x = ttk.Scrollbar(data_window, orient=tk.HORIZONTAL, command=text_widget.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        text_widget.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 데이터 표시
        dataset = self.predefined_datasets[selected]
        text_widget.insert(tk.END, f"데이터셋: {selected}\n")
        text_widget.insert(tk.END, f"총 데이터 수: {len(dataset['inputs'])}\n\n")
        text_widget.insert(tk.END, "입력값\t\t목표값\n")
        text_widget.insert(tk.END, "-" * 50 + "\n")
        
        for i, (inputs, targets) in enumerate(zip(dataset['inputs'], dataset['targets'])):
            text_widget.insert(tk.END, f"데이터 {i+1}:\n")
            text_widget.insert(tk.END, f"입력: {inputs}\n")
            text_widget.insert(tk.END, f"목표: {targets}\n")
            text_widget.insert(tk.END, "-" * 50 + "\n")
        
        # 텍스트 위젯을 읽기 전용으로 설정
        text_widget.configure(state='disabled')

# 메인 실행 코드
if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop() 