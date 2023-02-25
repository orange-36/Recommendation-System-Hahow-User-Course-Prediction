import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import csv
from tqdm import tqdm
import sys

user_haved_purchased_course = {}

def get_course_score_by_course_embedding(cosine_sim_sum, haved_courses_list, weight, cosine_sim):
    haved_courses_index_list = [ course2id_mapping[course_id] for course_id in haved_courses_list]

    for idx in haved_courses_index_list:
        sim_scores = list(enumerate(cosine_sim[idx]))
        for i in range(len(sim_scores)):
            cosine_sim_sum[i][1] += weight*sim_scores[i][1]

    return cosine_sim_sum

def get_recommendations_subgroup_by_course_similarity(user_id, similar_users_list, similar_user_weights_list, cosine_sim, top = 10):
    cosine_sim_sum = [ [i, 0] for i in range(len(course2id_mapping))]
    haved_courses_list = user_haved_purchased_course[user_id]
    haved_courses_index_list = [ course2id_mapping[course_id] for course_id in haved_courses_list]
    for idx in haved_courses_index_list:
        sim_scores = list(enumerate(cosine_sim[idx]))
        for i in range(len(sim_scores)):
            cosine_sim_sum[i][1] += sim_scores[i][1]

    sim_scores = sorted(cosine_sim_sum, key=lambda x: x[1], reverse=True)

    recommend_subgroups = []
    for i in range(len(sim_scores)):
        if len(recommend_subgroups) < top:
            if sim_scores[i][0] not in haved_courses_index_list:
                course_id = course_df['course_id'][sim_scores[i][0]]
                for subgroup in course2subgroups[course_id]:
                    if str(subgroup) not in recommend_subgroups:
                        recommend_subgroups.append(str(subgroup))

    return recommend_subgroups

def get_recommendations_course_by_course_embedding_and_similar_users(user_id, similar_users_list, similar_user_weights_list, cosine_sim, top = 10):
    cosine_sim_sum = [ [i, 0] for i in range(len(course2id_mapping))]
    
    if user_id in user_haved_purchased_course.keys():
        cosine_sim_sum = get_course_score_by_course_embedding(cosine_sim_sum, user_haved_purchased_course[user_id], 1, cosine_sim)

    for similar_user, similar_user_weight in zip(similar_users_list, similar_user_weights_list):
        cosine_sim_sum = get_course_score_by_course_embedding(cosine_sim_sum, user_haved_purchased_course[similar_user], (similar_user_weight)**2, cosine_sim)

    sim_scores = sorted(cosine_sim_sum, key=lambda x: x[1], reverse=True)
    recommend_indices = []
    magic_number = [665, 673, 664, 656]

    for i in range(len(sim_scores)):
        if i == 2:
            for magic in magic_number:
                if user_id in user_haved_purchased_course.keys():
                    if id2course_mapping[magic] not in user_haved_purchased_course[user_id]:
                        if magic not in recommend_indices: 
                            recommend_indices.append(magic)
                else:
                    if magic not in recommend_indices: 
                        recommend_indices.append(magic)  
        if len(recommend_indices) < top:
            if user_id in user_haved_purchased_course.keys():
                if id2course_mapping[sim_scores[i][0]] not in user_haved_purchased_course[user_id]:
                    if sim_scores[i][0] not in recommend_indices: 
                        recommend_indices.append(sim_scores[i][0])
            else:
                if sim_scores[i][0] not in recommend_indices: 
                        recommend_indices.append(sim_scores[i][0])
        else:
            break

    return course_df['course_id'].iloc[recommend_indices].tolist()

def get_recommendations_subgroup_by_course_embedding_and_similar_users(user_id, similar_users_list, similar_user_weights_list, cosine_sim, top = 10):
    cosine_sim_sum = [ [i, 0] for i in range(len(course2id_mapping))]

    if user_id in user_haved_purchased_course.keys():
        cosine_sim_sum = get_course_score_by_course_embedding(cosine_sim_sum, user_haved_purchased_course[user_id], 1, cosine_sim)
    for similar_user, similar_user_weight in zip(similar_users_list, similar_user_weights_list):
        cosine_sim_sum = get_course_score_by_course_embedding(cosine_sim_sum, user_haved_purchased_course[similar_user], (similar_user_weight)**2, cosine_sim)

    sim_scores = sorted(cosine_sim_sum, key=lambda x: x[1], reverse=True)

    recommend_subgroups = []
    magic_number = [665, 673, 664, 656]
    for magic in magic_number:
        if user_id in user_haved_purchased_course.keys():
            if id2course_mapping[magic]not in user_haved_purchased_course[user_id]:
                course_id = course_df['course_id'][magic]
                for subgroup in course2subgroups[course_id]:
                    if str(subgroup) not in recommend_subgroups:
                        recommend_subgroups.append(str(subgroup))
        else:
            course_id = course_df['course_id'][magic]
            for subgroup in course2subgroups[course_id]:
                if str(subgroup) not in recommend_subgroups:
                    recommend_subgroups.append(str(subgroup))    

    for i in range(len(sim_scores)):
        if len(recommend_subgroups) < top:
            if user_id in user_haved_purchased_course.keys():
                if id2course_mapping[sim_scores[i][0]]not in user_haved_purchased_course[user_id]:
                    course_id = course_df['course_id'][sim_scores[i][0]]
                    for subgroup in course2subgroups[course_id]:
                        if str(subgroup) not in recommend_subgroups:
                            recommend_subgroups.append(str(subgroup))
            else:
                course_id = course_df['course_id'][sim_scores[i][0]]
                for subgroup in course2subgroups[course_id]:
                    if str(subgroup) not in recommend_subgroups:
                        recommend_subgroups.append(str(subgroup))
        else:
            break

    return recommend_subgroups

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('There should have 4 arguments')
        print('argument 1 is test file path')
        print('argument 2 is output prediction file path')
        print('argument 3 is task type(Course or Topic')
        print('argument 4 is task type(Seen or Unseen')
        print('argument 5 is recommend method(Both_UserSim_and_CourseSim or Only_CourseSim, default is Both_UserSim_and_CourseSim')
        sys.exit()

    assert (sys.argv[3]=="Course") or (sys.argv[3]=="Topic") , 'argument 3 必須是 Course 或 Topic'
    assert (sys.argv[4]=="Seen") or (sys.argv[4]=="Unseen") , 'argument 4 必須是 Seen 或 Unseen'

    test_file_path = sys.argv[1]
    predictions_file_path = Path(sys.argv[2])
    print( f"Task : {sys.argv[3]} {sys.argv[4]}")
    if sys.argv[3]=="Course":
        similar_num = 150
        predict_column2 = "course_id"
        get_recommendations_function = get_recommendations_course_by_course_embedding_and_similar_users
    else:
        similar_num = 75
        predict_column2 = "subgroup"
        get_recommendations_function = get_recommendations_subgroup_by_course_embedding_and_similar_users

    if sys.argv[4]=="Seen":
        user_similar_path = 'similar_users/seen_user_similar_add_course_300.pickle'
    else:
        user_similar_path = 'similar_users/unseen_user_similar_add_course_300.pickle'

    if len(sys.argv) == 6:
        if sys.argv[5] == "Only_CourseSim" and sys.argv[3]=="Topic" and sys.argv[4]=="Seen":
            get_recommendations_function = get_recommendations_subgroup_by_course_similarity
            print("Only use course similarity")
    else:
        print("Use both user similarity and course similarity")

    if str(predictions_file_path.parent)!='.':
        Path(predictions_file_path.parent).mkdir(parents = True, exist_ok= True)

    with open('content_embeding/pure_context_jieba_name_target_chapters.pickle', 'rb') as f:
        pure_context_jieba_name_target_chapters = pickle.load(f)
    cosine_sim_pure_jieba = cosine_similarity(pure_context_jieba_name_target_chapters['Tfidf_matrix'], pure_context_jieba_name_target_chapters['Tfidf_matrix'])

    with open(user_similar_path, 'rb') as f:
        test_predict_similiar_users = pickle.load(f)


    users_df = pd.read_csv('./data/users.csv')
    course_df = pd.read_csv('./data/courses.csv')
    train_df = pd.read_csv('./data/train.csv')
    val_seen_df = pd.read_csv('./data/val_seen.csv')
    test_seen_df = pd.read_csv(test_file_path)
    val_unseen_df = pd.read_csv('./data/val_unseen.csv')
    subgroups_df = pd.read_csv('./data/subgroups.csv')

    id2course_mapping = course_df["course_id"].to_dict()
    course2id_mapping = {v : k for k, v in id2course_mapping.items()}

    subgroups2idx = {}
    for (id, name) in zip(subgroups_df["subgroup_id"], subgroups_df["subgroup_name"]):
        subgroups2idx.update({name: id})
    idx2subgroups = {v : k for k, v in subgroups2idx.items()}
    course2subgroups = {}
    for (course_id, sub_groups) in zip(course_df["course_id"], course_df["sub_groups"]):
        if pd.isnull(sub_groups):
            course2subgroups.update({course_id: [0]})
        else:
            course2subgroups.update({course_id: [ subgroups2idx[sub_group] for sub_group in sub_groups.split(',')]})

    id2user_mapping = users_df["user_id"].to_dict()
    user2id_mapping = {v : k for k, v in id2user_mapping.items()}

    train_user_haved_purchased_course = {}
    for seen_user_id, course_ids in zip(train_df["user_id"], train_df["course_id"]):
        course_id_list = course_ids.split(' ')
        train_user_haved_purchased_course[seen_user_id] = train_user_haved_purchased_course.setdefault(seen_user_id, []) + course_id_list
        user_haved_purchased_course[seen_user_id] = user_haved_purchased_course.setdefault(seen_user_id, []) + course_id_list

    val_seen_user_haved_purchased_course = {}
    for seen_user_id, course_ids in zip(val_seen_df["user_id"], val_seen_df["course_id"]):
        course_id_list = course_ids.split(' ')
        val_seen_user_haved_purchased_course[seen_user_id] = val_seen_user_haved_purchased_course.setdefault(seen_user_id, []) + course_id_list
        user_haved_purchased_course[seen_user_id] = user_haved_purchased_course.setdefault(seen_user_id, []) + course_id_list
        
    val_unseen_user_haved_purchased_course = {}
    for unseen_user_id, course_ids in zip(val_unseen_df["user_id"], val_unseen_df["course_id"]):
        course_id_list = course_ids.split(' ')
        val_unseen_user_haved_purchased_course[unseen_user_id] = val_unseen_user_haved_purchased_course.setdefault(unseen_user_id, []) + course_id_list
        user_haved_purchased_course[unseen_user_id] = user_haved_purchased_course.setdefault(unseen_user_id, []) + course_id_list

    test_predict_users = test_seen_df["user_id"].to_list()
    print(f"There are {len(test_predict_users)} users")
    with open(predictions_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", predict_column2])
        for user_id in tqdm(test_predict_users):
            predictions = get_recommendations_function(user_id, 
                        test_predict_similiar_users["similar_users"][user_id][:similar_num], test_predict_similiar_users["similar_user_weights"][user_id][:similar_num], cosine_sim_pure_jieba, 50)
            recommend = " ".join(predictions)
            writer.writerow([user_id, recommend])
