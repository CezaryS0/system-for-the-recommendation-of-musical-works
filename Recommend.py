from Recommendation.Recommendation import Recommendation

name = 'Öron Ignorera'
test_data = 'Train_Data/Test/slices'
rec = Recommendation()
distance_array,titles =  rec.generate_recommendations('Saved_Model/Model.h5',test_data,name)
rec.print_predictions(distance_array,titles)