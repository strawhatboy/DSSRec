import pandas as pd
from sklearn.preprocessing import LabelEncoder

USER_VIDEO_FILE = './data/rel_user_video.csv'
COURSE_VIDEO_FILE = './data/rel_course_video.csv'

def main():
    user_video = pd.read_csv(USER_VIDEO_FILE)
    course_video = pd.read_csv(COURSE_VIDEO_FILE)

    user_video = user_video.drop_duplicates(subset=['start_id', 'end_id'])
    course_video = course_video.drop_duplicates(subset=['start_id', 'end_id'])


    video_encoder = LabelEncoder()
    video_encoder.fit(pd.concat([user_video['end_id'], course_video['end_id']]))
    user_video['end_id'] = video_encoder.transform(user_video['end_id'])
    course_video['end_id'] = video_encoder.transform(course_video['end_id'])

    # user_video.to_csv('./data/rel_user_video_encoded.csv', index=False)
    # process user_video to sequence
    user_video = user_video[user_video.groupby('start_id')['end_id'].transform('count') > 10]
    user_video['end_id'] = LabelEncoder().fit_transform(user_video['end_id'])
    user_encoder = LabelEncoder()
    course_encoder = LabelEncoder()
    user_video['start_id'] = user_encoder.fit_transform(user_video['start_id'])
    course_video['start_id'] = course_encoder.fit_transform(course_video['start_id'])
    user_video_seq = user_video.groupby('start_id')['end_id'].apply(lambda x: ','.join(list(x.astype(str)))).reset_index()

    user_video_seq.rename(columns={'start_id': 'id'}, inplace=True)
    user_video_seq['id'] = LabelEncoder().fit_transform(user_video_seq['id'])
    user_video_seq.rename(columns={'end_id': 'video_ids'}, inplace=True)
    user_video_seq.to_csv('./data/MOOCCube.csv', index=False)
    
    course_video.rename(columns={'start_id': 'course_id'}, inplace=True)
    course_video.rename(columns={'end_id': 'video_id'}, inplace=True)
    course_video.to_csv('./data/MOOCCube_video2course.csv', index=False)

    print('user count: {}'.format(user_encoder.classes_.shape[0]))
    print('course count: {}'.format(course_encoder.classes_.shape[0]))
    print('video count: {}'.format(video_encoder.classes_.shape[0]))



if __name__ == '__main__':
    main()
