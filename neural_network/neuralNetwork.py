import numpy as np
import json
import os

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes  # 리스트로 여러 은닉층의 크기를 받음
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
        self.activations = [None] * (len(hidden_sizes) + 2)  # 입력층 + 은닉층들 + 출력층
        
        # 학습 히스토리 저장
        self.weight_history = []
        self.error_history = []
        
        # 저장 디렉토리 생성
        self.save_dir = 'neural_network/saved_states'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

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
        
        # Forward pass
        # 각 층의 출력값을 저장할 리스트
        layer_outputs = [[] for _ in range(len(self.weights) + 1)]
        layer_outputs[0] = inputs  # 입력층 출력
        
        # 순전파 수행
        current_input = inputs
        for i, weight in enumerate(self.weights):
            current_output = self.sigmoid(np.dot(current_input, weight))
            layer_outputs[i + 1] = current_output
            current_input = current_output
        
        outputs = layer_outputs[-1]  # 최종 출력
        
        # 역전파
        # 출력층 오차와 델타
        output_errors = targets - outputs
        deltas = [output_errors * self.sigmoid_derivative(outputs)]
        
        # 은닉층들의 오차와 델타
        for i in range(len(self.weights) - 1, 0, -1):
            error = np.dot(deltas[0], self.weights[i].T)
            delta = error * self.sigmoid_derivative(layer_outputs[i])
            deltas.insert(0, delta)
        
        # 가중치 업데이트
        for i in range(len(self.weights)):
            layer_input = layer_outputs[i]
            layer_delta = deltas[i]
            self.weights[i] += self.learning_rate * np.dot(layer_input.T, layer_delta)
        
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

    def save_state(self, epoch):
        """학습 상태를 파일로 저장"""
        state = {
            'epoch': epoch,
            'weights': [w.tolist() for w in self.weights],  # NumPy 배열을 리스트로 변환
            'error': float(self.error_history[-1]) if self.error_history else None  # float로 변환
        }
        
        filename = os.path.join(self.save_dir, f'network_state_epoch_{epoch}.json')
        with open(filename, 'w') as f:
            json.dump(state, f)

    def load_state(self, epoch):
        """저장된 학습 상태 불러오기"""
        filename = os.path.join(self.save_dir, f'network_state_epoch_{epoch}.json')
        with open(filename, 'r') as f:
            state = json.load(f)
            
        # 리스트를 다시 NumPy 배열로 변환
        self.weights = [np.array(w) for w in state['weights']]
        return state['error']

    def reset_weights(self):
        """가중치 초기화"""
        self.weights = []
        prev_size = self.input_size
        for hidden_size in self.hidden_sizes:
            self.weights.append(np.random.randn(prev_size, hidden_size) * 0.1)
            prev_size = hidden_size
        self.weights.append(np.random.randn(prev_size, self.output_size) * 0.1)
        self.activations = [None] * (len(self.hidden_sizes) + 2)
