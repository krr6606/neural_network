# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
import tkinter as tk
from tkinter import ttk
from neural_network.neuralNetwork import SimpleNeuralNetwork
import random

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("신경망 학습 시각화 시스템")
        
        # 신경망 초기화 (더 큰 구조)
        self.nn = SimpleNeuralNetwork(
            input_size=6,              # 6개의 입력 노드
            hidden_sizes=[8, 6, 4],    # 3개의 은닉층
            output_size=3,             # 3개의 출력 노드
            learning_rate=0.1
        )
        
        # 학습 데이터 저장
        self.batch_size = 4  # 배치 크기 추가
        self.training_data = []
        self.current_data_index = 0
        self.epoch_count = 0
        self.error_history = []
        self.total_epochs = 1000  # 전체 학습 에포크 수 설정
        
        # 미리 정의된 데이터셋
        self.predefined_datasets = {
            # 기본 논리 게이트
            "XOR": {
                "inputs": [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], 
                          [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]],
                "targets": [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]]
            },
            "AND": {
                "inputs": [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], 
                          [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]],
                "targets": [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]]
            },
            
            # 이진 패턴 인식
            "이진패턴_A": {
                "inputs": [
                    [1, 1, 1, 1, 1, 1],  # 모든 1
                    [0, 0, 0, 0, 0, 0],  # 모든 0
                    [1, 0, 1, 0, 1, 0],  # 교차 패턴
                    [0, 1, 0, 1, 0, 1],  # 반대 교차
                    [1, 1, 0, 0, 1, 1],  # 양끝 1
                    [0, 0, 1, 1, 0, 0],  # 중간 1
                ],
                "targets": [[1, 0, 0], [0, 1, 0], [1, 1, 0], 
                           [0, 0, 1], [1, 0, 1], [0, 1, 1]]
            },
            
            # 연속적 패턴
            "연속패턴": {
                "inputs": [
                    [1, 1, 1, 0, 0, 0],  # 왼쪽 절반
                    [0, 0, 0, 1, 1, 1],  # 오른쪽 절반
                    [1, 1, 0, 0, 1, 1],  # 양끝 두개씩
                    [0, 1, 1, 1, 1, 0],  # 중간 넷
                    [1, 0, 0, 0, 0, 1],  # 양끝만
                    [0, 1, 1, 1, 1, 0],  # 중간만
                ],
                "targets": [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [0, 1, 1], [1, 0, 1]]
            },
            
            # 대칭 패턴
            "대칭패턴": {
                "inputs": [
                    [1, 0, 0, 0, 0, 1],  # 완전 대칭
                    [0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                    [1, 1, 0, 0, 1, 1],
                    [0, 1, 1, 1, 1, 0],
                    [1, 0, 1, 1, 0, 1]
                ],
                "targets": [[1, 1, 1], [1, 0, 1], [0, 1, 1],
                           [1, 1, 0], [0, 0, 1], [1, 0, 0]]
            },
            
            # 복합 패턴
            "복합패턴": {
                "inputs": [
                    [1, 1, 0, 0, 1, 1],  # 대칭 + 연속
                    [0, 1, 1, 1, 1, 0],  # 중간 집중
                    [1, 0, 1, 1, 0, 1],  # 교차 대칭
                    [1, 1, 1, 0, 0, 0],  # 편향 분포
                    [0, 0, 1, 1, 1, 1],  # 반대 편향
                    [1, 0, 0, 1, 1, 0],  # 불규칙
                    [0, 1, 1, 0, 0, 1],  # 다른 불규칙
                    [1, 1, 0, 1, 0, 1]   # 복합 불규칙
                ],
                "targets": [[1, 1, 1], [0, 1, 1], [1, 0, 1],
                           [1, 1, 0], [0, 1, 0], [1, 0, 0],
                           [0, 0, 1], [1, 1, 1]]
            },
            
            # 순차적 패턴
            "순차패턴": {
                "inputs": [
                    [1, 0, 0, 0, 0, 0],  # 한칸씩 이동
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]
                ],
                "targets": [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            },
            
            # 랜덤 패턴 (미리 생성된 랜덤 데이터)
            "랜덤패턴": {
                "inputs": [
                    [round(random.random()) for _ in range(6)] for _ in range(10)
                ],
                "targets": [
                    [round(random.random()) for _ in range(3)] for _ in range(10)
                ]
            }
        }
        
        # GUI 레이아웃 설정
        self.setup_gui()
        self.setup_error_plot()  # 오차 그래프 설정 추가
        
        # 애니메이션 시작
        self.is_training = False
        self.frame_count = 0
        self.ani = None
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
        self.batch_var = tk.IntVar(value=4)
        ttk.Entry(batch_frame, textvariable=self.batch_var, width=5).pack(side=tk.LEFT, padx=5)
        
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
        
        # 데이터셋 선택 섹션 추가
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
        
        # 우측 패널 (시각화)
        right_panel = ttk.LabelFrame(main_container, text="신경망 구조")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.fig = plt.Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        """학습률 업데이트"""
        try:
            new_lr = self.lr_var.get()
            if new_lr > 0:
                self.nn.learning_rate = new_lr
                self.status_text.set(f"학습률이 {new_lr:.4f}로 변경됨")
        except:
            self.status_text.set("학습률 변경 실패")

    def toggle_training(self):
        self.is_training = not self.is_training
        # 버튼 텍스트 업데이트
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
            for _ in range(10):
                self.epoch_count += 1
                self.epoch_text.set(f"에포크: {self.epoch_count}")
                
                error = 0
                if self.train_mode.get() == "single":
                    # 단일 입력 학습
                    inputs = np.array([[var.get() for var in self.input_vars]])
                    targets = np.array([[var.get() for var in self.target_vars]])
                    error = self.nn.train(inputs, targets)
                    # 학습 후 순전파 수행하여 활성화 값 업데이트
                    self.nn.forward(inputs)
                else:
                    # 전체 데이터 배치 학습
                    if len(self.training_data) > 0:
                        try:
                            batch_size = min(self.batch_var.get(), len(self.training_data))
                            # 배치 데이터 준비
                            batch_indices = np.random.choice(len(self.training_data), batch_size, replace=False)
                            batch_inputs = []
                            batch_targets = []
                            
                            for idx in batch_indices:
                                inputs, targets = self.training_data[idx]
                                batch_inputs.append(inputs)
                                batch_targets.append(targets)
                            
                            # 배치 학습
                            inputs = np.array(batch_inputs, dtype=np.float32)
                            targets = np.array(batch_targets, dtype=np.float32)
                            error = self.nn.train(inputs, targets)
                            
                            # 시각화를 위해 현재 배치의 첫 번째 데이터로 순전파
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
                if self.epoch_count >= self.total_epochs_var.get():
                    self.is_training = False
                    self.train_button.config(text="학습 시작")
                    self.status_text.set("학습 완료")
                    break
            
            self.draw_network()

    def start_animation(self):
        """애니메이션 시작"""
        if self.ani is not None:
            self.ani.event_source.stop()
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.animate, 
            interval=50,
            blit=False
        )

    def save_current_data(self):
        """현재 입력값과 목표값을 학습 데이터로 저장"""
        inputs = [var.get() for var in self.input_vars]
        targets = [var.get() for var in self.target_vars]
        self.training_data.append((inputs, targets))
        self.status_text.set(f"데이터 {len(self.training_data)}개 저장됨")
        print(f"데이터 포인트 {len(self.training_data)}개 저장됨")

    def clear_training_data(self):
        """저장된 학습 데이터 초기화"""
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
        self.epoch_text.set("에포크: 0")
        self.error_text.set("오차: 0.0")
        
        # 시각화 업데이트
        self.draw_network()
        self.update_error_plot()

    def update_error_plot(self):
        """오차 그래프 업데이트"""
        self.error_ax.clear()
        if self.error_history:
            self.error_ax.plot(self.error_history[-100:])  # 최근 100개 포인트만 표시
            self.error_ax.set_title('학습 오차')
            self.error_ax.set_xlabel('에포크')
            self.error_ax.set_ylabel('오차')
            self.error_canvas.draw()

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
            if selected in self.predefined_datasets:
                dataset = self.predefined_datasets[selected]
                
                # 기존 학습 상태 초기화 여부 확인
                if self.epoch_count > 0:
                    if tk.messagebox.askyesno("초기화 확인", 
                        "현재 학습 상태를 초기화하시겠습니까?\n'아니오'를 선택하면 현재 가중치를 유지합니다."):
                        self.nn.reset_weights()
                        self.error_history = []
                
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

def main():
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 