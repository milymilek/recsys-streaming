from recsys_streaming_ml.data.utils import *

def test_build_reverse_feature_maps():
    fm = {"feature1": {"user1": 0, "user2": 1, "user3": 2}, "feature2": {"user1": 9, "user2": 8, "user3": 7}}
    result = build_reverse_feature_maps(fm)
    expected_result = {"feature1": {0: "user1", 1: "user2", 2: "user3"}, "feature2": {9: "user1", 8: "user2", 7: "user3"}}
    assert result == expected_result


