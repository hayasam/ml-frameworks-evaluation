import unittest
import params_checker

class SimplisticTest(unittest.TestCase):

    def test_net_diff_is_empty(self):
        differentiator = params_checker.TrainingDiff()
        differentiator.init_models()
        differentiator.do_net_diff()

        with open(differentiator.net_diff_name, 'r') as file:
            text = file.read().replace('\n', '')

        self.assertTrue(text == "")

    def test_vgg_diff_is_empty(self):
        differentiator = params_checker.TrainingDiff()
        differentiator.init_models()
        differentiator.do_vgg_diff()

        with open(differentiator.vgg_diff_name, 'r') as file:
            text = file.read().replace('\n', '')

        self.assertTrue(text == "")
        
    def test_alex_diff_is_empty(self):
        differentiator = params_checker.TrainingDiff()
        differentiator.init_models()
        differentiator.do_alex_diff()

        with open(differentiator.alex_diff_name, 'r') as file:
            text = file.read().replace('\n', '')

        self.assertTrue(text == "")

    # bonus TODO test nbr params ?

    # TODO improve the attributes check (generator comparison with a mocked model/generator)
    def test_model_loader(self):
        differentiator = params_checker.TrainingDiff()
        differentiator.init_models()
        print(differentiator.buggy_net.modules())
        self.assertIsNotNone(differentiator.buggy_net)

    def test_if_same_str_params(self):
        differentiator = params_checker.TrainingDiff()
        differentiator.init_models()
        buggy_str_params = differentiator.find_str_params(differentiator.buggy_alex.alex_model.modules())
        corrected_str_params = differentiator.find_str_params(differentiator.corrected_alex.alex_model.modules())
        self.assertEqual(buggy_str_params, corrected_str_params)


if __name__ == "__main__":
    unittest.main()
