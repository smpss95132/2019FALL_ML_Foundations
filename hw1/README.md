# content
* problem6.py
* problem7.py
* problem8.py


## problem6.py
1. 載入資料並分成x及y兩部分
2. 設定x~0~為1及repeat次數為1126
3. 在開始PLA前進行洗牌的動作
4. 設定初始值
    * epoch_count=0   (計算更新weight的次數)
    * current_example=0 (目前檢查的example)
    * next_example=0  (下一個將要檢查的example)
    * finish_checking=-1 (最後一個讓weight被更新的example,故當  current_example==finish_checking時,代表已經檢查完一輪且所有example皆分類正確)
    * finish_flag=False (當此值為True時,跳出迴圈)
5. PLA algo
    1. 計算內積
    2. 檢查是否分類正確:
       正確=>檢查是否current_example==finish_checking,若是則設定finish_flag為True(PLA演算法完成)
       錯誤=>更新weight並將current_example assign給finish_checking
      
## problem7.py
* 大致上架構同problem6.py但多了以下改動
    1. evaluate_error_rate(feature,label,weight):用以計算給定weight在test set上的error rate
    2. current_weight:用以記錄截止至當前最佳的weight值
    3. update_time:當weight更新次數等於此值時,跳出while迴圈(結束pocket algo)
    
## peoblem8.py
* 同problem7.py只是沒有記錄當前最佳weight而已
    
    
    
