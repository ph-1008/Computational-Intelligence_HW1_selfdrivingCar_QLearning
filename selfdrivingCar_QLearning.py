import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import numpy as np
import os


class AutonomousCarSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("自走車模擬器 - Q-Learning")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- 以下部分與原程式相同的部分 ---
        # 增加標籤「訓練回合數:」(原本是選擇訓練檔案，這裡改成設定 Q-Learning 訓練回合數)
        training_label = tk.Label(self.root, text="訓練回合數:")
        training_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        # 使用 Entry 來輸入訓練回合數
        self.episodes_var = tk.StringVar(value="700")
        self.episodes_entry = tk.Entry(self.root, textvariable=self.episodes_var, width=10)
        self.episodes_entry.grid(row=0, column=0, padx=100, pady=5, sticky='w')

        # 訓練按鈕(原程式有 Training 按鈕)
        self.train_button = tk.Button(root, text="Training", command=self.train_model)
        self.train_button.grid(row=0, column=1, padx=0, pady=5)

        # 運行按鈕(原程式有 Run 按鈕)
        self.run_button = tk.Button(root, text="Run", command=self.run_simulation)
        self.run_button.grid(row=0, column=2, padx=0, pady=5)

        # --- 以下部分與原程式相同 ---
        # 繪圖區域：建立畫布來顯示模擬狀態
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)

        # 初始化軌道和車子 (與原程式相同)
        self.track_points = []
        self.start_position = None

        # 加載並顯示軌道和車輛 (原程式使用 load_track() 與 draw_track())
        self.load_track()
        self.draw_track()

        # --- 以下部分為 Q-Learning 模型相關內容，與原程式 RBFN 模型不同 ---
        # 先設定 Q-table 檔案名稱，若存在則直接載入
        self.qtable_filename = "qtable.npy"
        # 初始化 Q-Learning 參數與 Q 表
        self.q_table = np.zeros((3, 3, 3, 5))  # 狀態維度：3個感測器各3區間，動作空間：5個離散方向盤角度
        if os.path.exists(self.qtable_filename):
            self.q_table = np.load(self.qtable_filename)
            print("已找到 Q-table 檔案，直接載入 Q-table")

        self.alpha = 0.1  # 學習率 (改動：Q-Learning 更新用到)
        self.gamma = 0.9  # 折扣因子 (改動：Q-Learning 更新用到)
        self.epsilon = 1.0  # 探索率 (改動：採用 ε-greedy 策略)
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995  # 每回合衰減 (改動：控制探索率的衰減)

        # --- 以下部分為動作映射，改用 Q-Learning 之動作離散化 ---
        # 動作映射：動作索引對應的方向盤角度（單位：度），範圍 -40° ~ +40° 分成 5 個選項
        self.action_mapping = [-30, -15, 0, 15, 30]

        # 顯示感測器距離的標籤 (原程式相同)
        self.distance_label = tk.Label(root, text="距離: ", justify=tk.LEFT)
        self.distance_label.grid(row=2, column=0, columnspan=3, sticky='w', padx=5)

    def on_closing(self):
        # 與原程式相同的關閉窗口程式碼
        self.root.quit()
        self.root.destroy()
        sys.exit()

    def load_track(self):
        # 與原程式相同：讀取軌道座標點檔案，第一行為起始位置，其餘行為軌道邊界節點
        file_path = "./軌道座標點.txt"
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # 讀取起始位置 (原程式相同)
            start_x, start_y, start_angle = map(float, lines[0].split(','))
            self.start_position = (start_x, start_y, start_angle)
            # 從第四行開始讀取軌道邊界節點 (原程式相同)
            for line in lines[3:]:
                x, y = map(float, line.split(','))
                self.track_points.append((x, y))
            self.track_points.append(self.track_points[0])  # 封閉軌道

    def draw_track(self, show_start_position=True):
        # 與原程式相同：清除之前的圖形，並繪製軌道、終點區域、起點橫線及起始位置標記
        self.ax.clear()
        self.ax.grid(True)
        if self.track_points:
            x, y = zip(*self.track_points)
            self.ax.plot(x, y, 'b-')
            self.ax.set_xlabel("X axis")
            self.ax.set_ylabel("Y axis")
            self.ax.axis('equal')
        # 繪製終點區域（根據作業說明）
        end_zone_x = [18, 30, 30, 18, 18]
        end_zone_y = [40, 40, 37, 37, 40]
        self.ax.fill(end_zone_x, end_zone_y, 'lightcoral', alpha=0.5)
        # 繪製起點橫線
        self.ax.plot([-7, 7], [0, 0], 'k-', linewidth=2)
        if show_start_position and self.start_position:
            start_x, start_y, start_angle = self.start_position
            car_circle = plt.Circle((start_x, start_y), 3, color='red')
            self.ax.add_patch(car_circle)
            self.ax.annotate(f"starting angle: {start_angle}°", (start_x, start_y),
                             textcoords="offset points", xytext=(0, 10), ha='center')
        self.canvas.draw()

    def discretize_distance(self, d):
        # --- [新增] 將 0~50 的感測器距離劃分為3個區間：近、中、遠 ---
        if d < 6:
            return 0  # 近
        elif d < 14:
            return 1  # 中
        else:
            return 2  # 遠

    def get_state(self, distances):
        # --- [新增] 將3個感測器的連續距離轉換為離散狀態 ---
        return (self.discretize_distance(distances[0]),
                self.discretize_distance(distances[1]),
                self.discretize_distance(distances[2]))

    def calculate_distances(self, car_pos, car_angle):
        # 與原程式相同：計算車子與軌道邊界的感測器距離
        x, y = car_pos
        angle_rad = np.radians(car_angle)
        angles = [0, -45, 45]  # 前、右45度、左45度 (原程式相同)
        distances = []
        for sensor_angle in angles:
            abs_angle = angle_rad + np.radians(sensor_angle)
            max_distance = 50
            ray_points = np.array([[x, y],
                                   [x + max_distance * np.cos(abs_angle),
                                    y + max_distance * np.sin(abs_angle)]])
            min_dist = max_distance
            for i in range(len(self.track_points) - 1):
                wall = np.array([self.track_points[i], self.track_points[i + 1]])
                x1, y1 = ray_points[0]
                x2, y2 = ray_points[1]
                x3, y3 = wall[0]
                x4, y4 = wall[1]
                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denominator == 0:
                    continue
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersection_x = x1 + t * (x2 - x1)
                    intersection_y = y1 + t * (y2 - y1)
                    dist = np.sqrt((x - intersection_x) ** 2 + (y - intersection_y) ** 2)
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        return distances

    def check_end_zone(self, pos):
        # 與原程式相同：檢查是否到達終點區域
        x, y = pos
        return (18 <= x <= 30) and (37 <= y <= 40)

    def check_collision(self, pos):
        # 與原程式相同：檢查是否碰撞軌道邊界
        x, y = pos
        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i + 1]
            distance = self.point_to_line_distance(pos, p1, p2)
            if distance < 3:  # 假設車體半徑為3
                return True
        return False

    def point_to_line_distance(self, point, line_start, line_end):
        # 與原程式相同：計算點到線段的距離
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if l2 == 0:
            return np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2))
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        return np.sqrt((x - projection_x) ** 2 + (y - projection_y) ** 2)

    def train_model(self):
        """利用 Q-Learning 進行模型訓練
           [改動] 原程式使用 RBFN 模型，這裡改成 Q-Learning 的訓練步驟
           	並在 qlearning_training_log.txt 中記錄 step, epsilon 與 終止原因（到達終點/碰撞邊界/超過最大步數），
            訓練結束後自動儲存 Q-table。"""
        print("開始 Q-Learning 訓練")
        try:
            episodes = int(self.episodes_var.get())
        except:
            episodes = 700
        max_steps = 500  # 每個回合的最大步數
        training_log = []  # 紀錄每回合的總回饋

        # 重置 Q 表與探索率 (與原程式初始化部分類似，但採用 Q-Learning)
        self.q_table = np.zeros((3, 3, 3, 5))
        self.epsilon = 1.0

        for ep in range(episodes):
            # 重置環境，從起點出發 (與原程式 run_simulation 部分類似)
            current_pos = (self.start_position[0], self.start_position[1])
            car_angle = self.start_position[2]  # 單位：度
            phi = np.radians(car_angle)
            total_reward = 0
            steps = 0
            done = False
            termination_reason = ""  # 紀錄回合結束原因

            while not done and steps < max_steps:
                # 取得感測器數值並離散化成狀態
                sensor_distances = self.calculate_distances(current_pos, np.degrees(phi))
                state = self.get_state(sensor_distances)

                # --- [新增] 依 ε-greedy 原則選擇動作 ---
                if np.random.rand() < self.epsilon:
                    action_index = np.random.randint(0, 5)
                else:
                    action_index = np.argmax(self.q_table[state[0], state[1], state[2], :])
                steering_angle = self.action_mapping[action_index]

                # --- [新增] 依據 Q-Learning 的動作更新公式，用運動方程式更新車體位置與朝向 ---
                theta_rad = np.radians(steering_angle)
                new_x = current_pos[0] + np.cos(phi + theta_rad) + np.sin(theta_rad) * np.sin(phi)
                new_y = current_pos[1] + np.sin(phi + theta_rad) - np.sin(theta_rad) * np.cos(phi)
                new_pos = (new_x, new_y)
                # 更新車體朝向 (確保輸入 arcsin 參數在合法範圍內)
                val = (2 * np.sin(theta_rad)) / 6
                val = np.clip(val, -1, 1)
                new_phi = phi - np.arcsin(val)

                # --- [新增] 設計 Reward 與終止條件 ---
                if self.check_collision(new_pos):
                    reward = -80  # 碰撞邊界獎懲
                    termination_reason = "crashed"
                    done = True
                elif self.check_end_zone(new_pos):
                    reward = 150  # 到達終點大獎勵
                    termination_reason = "Goal"
                    done = True
                else:
                    reward = -0.1  # 每走一步的負獎勵鼓勵短路徑
                total_reward += reward

                # 取得下一狀態
                sensor_distances_next = self.calculate_distances(new_pos, np.degrees(new_phi))
                next_state = self.get_state(sensor_distances_next)

                # --- [新增] Q-Learning 更新公式 ---
                best_next_q = np.max(self.q_table[next_state[0], next_state[1], next_state[2], :])
                current_q = self.q_table[state[0], state[1], state[2], action_index]
                self.q_table[state[0], state[1], state[2], action_index] = current_q + \
                                                                           self.alpha * (
                                                                                       reward + self.gamma * best_next_q - current_q)

                # 更新狀態、位置與朝向 (與原程式類似)
                current_pos = new_pos
                phi = new_phi
                steps += 1

            # 若回合因步數達到上限而結束，則設定終止原因
            if termination_reason == "":
                termination_reason = "out of max_steps"

            # 每回合結束後衰減 epsilon (用於 ε-greedy 策略)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            training_log.append((ep + 1, total_reward, steps, self.epsilon, termination_reason))
            if (ep + 1) % 10 == 0:
                print(
                    f"Episode {ep + 1}/{episodes}: Reward = {total_reward:.3f}, Steps = {steps}, Epsilon = {self.epsilon:.3f}, {termination_reason}")

        # 將訓練 log 存檔以便重現實驗結果
        with open("qlearning_training_log.txt", "w") as f:
            for log in training_log:
                episode_number, total_reward, steps, epsilon_val, termination_reason = log
                f.write(f"Episode {episode_number}/{episodes}: Reward = {total_reward:.3f}, Steps = {steps}, Epsilon = {epsilon_val:.3f}, {termination_reason}\n")
        # 儲存 Q-table 至檔案
        np.save(self.qtable_filename, self.q_table)
        print("Q-Learning 訓練完成！")
        print(f"Q-table 已保存到 {self.qtable_filename}")

    def run_simulation(self):
        """利用訓練後的 Q 表進行模擬並畫出軌跡
           [改動] 原程式使用 RBFN 模型，此處改用 Q-Learning 模型依 Q 表選擇最佳動作"""
        if self.q_table is None or np.all(self.q_table == 0):
            print("模型尚未訓練！")
            return

        print("開始 Q-Learning 模型模擬")
        self.current_pos = (self.start_position[0], self.start_position[1])
        car_angle = self.start_position[2]
        phi = np.radians(car_angle)
        trajectory_x = [self.current_pos[0]]
        trajectory_y = [self.current_pos[1]]
        track_records = []
        b = 6  # 車輛軸距 (與原程式相同)

        self.draw_track(show_start_position=False)

        while True:
            sensor_distances = self.calculate_distances(self.current_pos, np.degrees(phi))
            distance_str = f'前: {sensor_distances[0]:.2f}, 右: {sensor_distances[1]:.2f}, 左: {sensor_distances[2]:.2f}'
            self.distance_label.config(text=distance_str)

            state = self.get_state(sensor_distances)
            # --- [新增] 從 Q 表中選擇最佳動作 (無隨機性) ---
            action_index = np.argmax(self.q_table[state[0], state[1], state[2], :])
            theta = self.action_mapping[action_index]

            # 紀錄當前狀態與選擇動作 (與原程式記錄方式相似)
            record = [self.current_pos[0], self.current_pos[1],
                      sensor_distances[0], sensor_distances[1], sensor_distances[2], theta]
            track_records.append(record)

            theta_rad = np.radians(theta)
            new_x = self.current_pos[0] + np.cos(phi + theta_rad) + np.sin(theta_rad) * np.sin(phi)
            new_y = self.current_pos[1] + np.sin(phi + theta_rad) - np.sin(theta_rad) * np.cos(phi)
            new_pos = (new_x, new_y)
            val = (2 * np.sin(theta_rad)) / 6
            val = np.clip(val, -1, 1)
            new_phi = phi - np.arcsin(val)

            self.current_pos = new_pos
            phi = new_phi

            trajectory_x.append(self.current_pos[0])
            trajectory_y.append(self.current_pos[1])

            self.draw_track(show_start_position=False)
            self.ax.plot(trajectory_x, trajectory_y, 'r-')
            car_circle = plt.Circle(self.current_pos, 3, color='red')
            self.ax.add_patch(car_circle)
            direction_length = 5
            direction_end = (self.current_pos[0] + direction_length * np.cos(phi),
                             self.current_pos[1] + direction_length * np.sin(phi))
            self.ax.plot([self.current_pos[0], direction_end[0]],
                         [self.current_pos[1], direction_end[1]], 'k-', linewidth=1)
            self.canvas.draw()

            if self.check_end_zone(self.current_pos):
                print("到達終點！")
                self.save_track_records(track_records)
                break
            if self.check_collision(self.current_pos):
                print("碰撞邊界！")
                break

            self.root.update()
            self.root.after(20)

    def save_track_records(self, records):
        """保存模擬過程記錄到檔案 (與原程式相同)"""
        filename = "qlearning_track.txt"
        filepath = f"./{filename}"
        with open(filepath, 'w') as f:
            for record in records:
                formatted_record = " ".join([f"{x:.7f}" for x in record])
                f.write(formatted_record + "\n")
        print(f"移動記錄已保存到 {filename}")


if __name__ == "__main__":
    # 創建主窗口 (與原程式相同)
    root = tk.Tk()
    app = AutonomousCarSimulator(root)
    root.mainloop()
