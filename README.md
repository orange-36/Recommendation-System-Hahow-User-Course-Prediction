# Content-Based Recommendation: Hahow Course/Topic Prediction 
![Rank Image](https://user-images.githubusercontent.com/69178839/221357423-18ab7489-a34b-4b1b-b6a2-d1721120e586.png)
![Rank Image](https://user-images.githubusercontent.com/69178839/221357414-60c58e0b-4c93-46bc-a7c9-e4b6b7219d83.png)
[\[2022 ADL Final Hahow Seen User Course Prediction\]](https://www.kaggle.com/competitions/2022-adl-final-hahow-seen-user-course-prediction)  
[\[2022 ADL Final Hahow Unseen User Course Prediction\]](https://www.kaggle.com/competitions/2022-adl-final-hahow-unseen-user-course-prediction)  
[\[2022 ADL Final Hahow Seen User Topic Prediction\]](https://www.kaggle.com/competitions/2022-adl-final-hahow-seen-user-topic-prediction)  
[\[2022 ADL Final Hahow Unseen User Topic Prediction\]](https://www.kaggle.com/competitions/2022-adl-final-hahow-unseen-user-topic-prediction) 
## Content-Based Method
* 首先，先準備好`data`資料夾(裡面包含原始的users.csv、courses.csv、train.csv、val_seen.csv、val_unseen.csv、subgroups.csv等資料)、`content_embeding`資料夾(
    裡面包含了運用TF-IDF所取得之課程向量，透過執行`course_embedding.ipynb`取得)以及`similar_users`資料夾(裡面包含運用TF-IDF得使用者相似後，和每位test user最相近的300位user，透過執行`user_similarity_and_weights.ipynb`取得)。
    接著執行以下確認好環境後再根據任務執行不同程式碼:
    ```bash
    pip install -r content_based_requirements.txt
    ```
* 接著執行recommend_course_by_content.py，其參數如下:
    1.test的檔案，裡面每列包含受推薦的user_id
    2.output prediction檔案的路徑名稱，以.csv結尾
    3.任務名稱1，為`"Course"`或是`"Topic"`
    4.任務名稱2，為`"Seen"`或是`"Unseen"`
    5.實作推薦課程的方式(選填)，為`"Both_UserSim_and_CourseSim"`或是`"Only_CourseSim"`，預設為`"Both_UserSim_and_CourseSim"`。
    `"Both_UserSim_and_CourseSim"`代表利用和使用者相似的使用者以及課程相似度來推薦；`"Only_CourseSim"`代表只運用過去課程的相似課程來推薦。
    Seen Coures、Unseen Course、Unseen Topic這三個任務使用`"Both_UserSim_and_CourseSim"`會有最好的成績；而Seen Topic則是`"Only_CourseSim"`
    ```bash=
    python3.9 recommend_course_by_content.py {test的檔案，裡面每列包含受推薦的user_id} {output prediction檔案的路徑名稱，以.csv結尾} {任務名稱1，為"Course"或是"Topic"} {任務名稱2，為"Seen"或"Unseen"} {推薦的方式，為"Both_UserSim_and_CourseSim"或"Only_CourseSim"}
    ```
* 執行範例:
    ### Seen Coures
    ```bash=
    python3.9 recommend_course_by_content.py data/test_seen.csv predictions/CourseSeen.csv "Course" "Seen"
    ```
    ### Unseen Course
    ```bash=
    python3.9 recommend_course_by_content.py data/test_unseen.csv predictions/CourseUnseen.csv "Course" "Unseen"
    ```
    ### Seen Topic
    ```bash=
    python3.9 recommend_course_by_content.py data/test_seen_group.csv predictions/TopicSeen.csv "Topic" "Seen" "Only_CourseSim"
    ```
    ### Unseen Topic
    ```bash=
    python3.9 recommend_course_by_content.py data/test_unseen.csv predictions/TopicUnseen.csv "Topic" "Unseen"
