import unittest
from skExSTraCS import ExSTraCS,StringEnumerator
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath("test_ExSTraCS.py"))
if THIS_DIR[-4:] == 'test': #Patch that ensures testing from Scikit not test directory
    THIS_DIR = THIS_DIR[:-5]

class test_ExSTrCS(unittest.TestCase):

    #learning_iterations
    def testParamLearningIterationsNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(learning_iterations="hello")
        self.assertTrue("learning_iterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterationsInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(learning_iterations=3.3)
        self.assertTrue("learning_iterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterationsInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(learning_iterations=-2)
        self.assertTrue("learning_iterations param must be nonnegative integer" in str(context.exception))

    def testParamLearningIterations(self):
        clf = ExSTraCS(learning_iterations=2000)
        self.assertEqual(clf.learning_iterations,2000)

    #N
    def testParamNNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(N="hello")
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamNInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(N=3.3)
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamNInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(N=-2)
        self.assertTrue("N param must be nonnegative integer" in str(context.exception))

    def testParamN(self):
        clf = ExSTraCS(N=2000)
        self.assertEqual(clf.N,2000)

    #nu
    def testNuInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(nu="hi")
        self.assertTrue("nu param must be float" in str(context.exception))

    def testNu1(self):
        clf = ExSTraCS(nu = -1)
        self.assertEqual(clf.nu,-1)

    def testNu2(self):
        clf = ExSTraCS(nu = 3)
        self.assertEqual(clf.nu,3)

    def testNu3(self):
        clf = ExSTraCS(nu = 1.2)
        self.assertEqual(clf.nu,1.2)

    #chi
    def testParamChiInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(chi="hello")
        self.assertTrue("chi param must be float from 0 - 1" in str(context.exception))

    def testParamChiInv2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(chi=3)
        self.assertTrue("chi param must be float from 0 - 1" in str(context.exception))

    def testParamChiInv3(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(chi=-1.2)
        self.assertTrue("chi param must be float from 0 - 1" in str(context.exception))

    def testParamChi1(self):
        clf = ExSTraCS(chi=0)
        self.assertEqual(clf.chi,0)

    def testParamChi2(self):
        clf = ExSTraCS(chi=0.3)
        self.assertEqual(clf.chi,0.3)

    def testParamChi3(self):
        clf = ExSTraCS(chi=1)
        self.assertEqual(clf.chi,1)

    #mu
    def testParamUpsilonInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(mu="hello")
        self.assertTrue("mu param must be float from 0 - 1" in str(context.exception))

    def testParamUpsilonInv2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(mu=3)
        self.assertTrue("mu param must be float from 0 - 1" in str(context.exception))

    def testParamUpsilonInv3(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(mu=-1.2)
        self.assertTrue("mu param must be float from 0 - 1" in str(context.exception))

    def testParamUpsilon1(self):
        clf = ExSTraCS(mu=0)
        self.assertEqual(clf.mu,0)

    def testParamUpsilon2(self):
        clf = ExSTraCS(mu=0.3)
        self.assertEqual(clf.mu,0.3)

    def testParamUpsilon3(self):
        clf = ExSTraCS(mu=1)
        self.assertEqual(clf.mu,1)

    #theta_GA
    def testParamThetaGAInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_GA="hello")
        self.assertTrue("theta_GA param must be nonnegative float" in str(context.exception))

    def testParamThetaGAInv3(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_GA=-1.2)
        self.assertTrue("theta_GA param must be nonnegative float" in str(context.exception))

    def testParamThetaGA1(self):
        clf = ExSTraCS(theta_GA=0)
        self.assertEqual(clf.theta_GA,0)

    def testParamThetaGA2(self):
        clf = ExSTraCS(theta_GA=1)
        self.assertEqual(clf.theta_GA,1)

    def testParamThetaGA3(self):
        clf = ExSTraCS(theta_GA=4.3)
        self.assertEqual(clf.theta_GA,4.3)

    #theta_sub
    def testParamThetaSubInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_sub="hello")
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_sub=2.3)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv3(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_sub=-1.2)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSubInv4(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_sub=-5)
        self.assertTrue("theta_sub param must be nonnegative integer" in str(context.exception))

    def testParamThetaSub1(self):
        clf = ExSTraCS(theta_sub=0)
        self.assertEqual(clf.theta_sub,0)

    def testParamThetaSub2(self):
        clf = ExSTraCS(theta_sub=5)
        self.assertEqual(clf.theta_sub,5)

    #acc_sub
    def testParamAccSubInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(acc_sub="hello")
        self.assertTrue("acc_sub param must be float from 0 - 1" in str(context.exception))

    def testParamAccSubInv2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(acc_sub=3)
        self.assertTrue("acc_sub param must be float from 0 - 1" in str(context.exception))

    def testParamAccSubInv3(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(acc_sub=-1.2)
        self.assertTrue("acc_sub param must be float from 0 - 1" in str(context.exception))

    def testParamAccSub1(self):
        clf = ExSTraCS(acc_sub=0)
        self.assertEqual(clf.acc_sub,0)

    def testParamAccSub2(self):
        clf = ExSTraCS(acc_sub=0.3)
        self.assertEqual(clf.acc_sub,0.3)

    def testParamAccSub3(self):
        clf = ExSTraCS(acc_sub=1)
        self.assertEqual(clf.acc_sub,1)

    #beta
    def testBetaInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(beta="hi")
        self.assertTrue("beta param must be float" in str(context.exception))

    def testBeta1(self):
        clf = ExSTraCS(beta = -1)
        self.assertEqual(clf.beta,-1)

    def testBeta2(self):
        clf = ExSTraCS(beta = 3)
        self.assertEqual(clf.beta,3)

    def testBeta3(self):
        clf = ExSTraCS(beta = 1.2)
        self.assertEqual(clf.beta,1.2)

    #delta
    def testDeltaInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(delta="hi")
        self.assertTrue("delta param must be float" in str(context.exception))

    def testDelta1(self):
        clf = ExSTraCS(delta = -1)
        self.assertEqual(clf.delta,-1)

    def testDelta2(self):
        clf = ExSTraCS(delta = 3)
        self.assertEqual(clf.delta,3)

    def testDelta3(self):
        clf = ExSTraCS(delta = 1.2)
        self.assertEqual(clf.delta,1.2)

    #init_fitness
    def testInitFitInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(init_fitness="hi")
        self.assertTrue("init_fitness param must be float" in str(context.exception))

    def testInitFit1(self):
        clf = ExSTraCS(init_fitness = -1)
        self.assertEqual(clf.init_fitness,-1)

    def testInitFit2(self):
        clf = ExSTraCS(init_fitness = 3)
        self.assertEqual(clf.init_fitness,3)

    def testInitFit3(self):
        clf = ExSTraCS(init_fitness = 1.2)
        self.assertEqual(clf.init_fitness,1.2)

    #fitness_reduction
    def testFitnessReductionInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(fitness_reduction="hi")
        self.assertTrue("fitness_reduction param must be float" in str(context.exception))

    def testFitnessReduction1(self):
        clf = ExSTraCS(fitness_reduction = -1)
        self.assertEqual(clf.fitness_reduction,-1)

    def testFitnessReduction2(self):
        clf = ExSTraCS(fitness_reduction = 3)
        self.assertEqual(clf.fitness_reduction,3)

    def testFitnessReduction3(self):
        clf = ExSTraCS(fitness_reduction = 1.2)
        self.assertEqual(clf.fitness_reduction,1.2)

    #theta_sel
    def testParamThetaSelInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_sel="hello")
        self.assertTrue("theta_sel param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSelInv2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_sel=3)
        self.assertTrue("theta_sel param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSelInv3(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(theta_sel=-1.2)
        self.assertTrue("theta_sel param must be float from 0 - 1" in str(context.exception))

    def testParamThetaSel1(self):
        clf = ExSTraCS(theta_sel=0)
        self.assertEqual(clf.theta_sel,0)

    def testParamThetaSel2(self):
        clf = ExSTraCS(theta_sel=0.3)
        self.assertEqual(clf.theta_sel,0.3)

    def testParamThetaSel3(self):
        clf = ExSTraCS(theta_sel=1)
        self.assertEqual(clf.theta_sel,1)

    #rule_specificity_limit
    def testParamRSLNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(rule_specificity_limit="hello")
        self.assertTrue("rule_specificity_limit param must be nonnegative integer or None" in str(context.exception))

    def testParamRSLInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(rule_specificity_limit=3.3)
        self.assertTrue("rule_specificity_limit param must be nonnegative integer or None" in str(context.exception))

    def testParamRSLInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(rule_specificity_limit=-2)
        self.assertTrue("rule_specificity_limit param must be nonnegative integer or None" in str(context.exception))

    def testParamRSL(self):
        clf = ExSTraCS(rule_specificity_limit=2000)
        self.assertEqual(clf.rule_specificity_limit,2000)

    def testParamRSL2(self):
        clf = ExSTraCS(rule_specificity_limit=None)
        self.assertEqual(clf.rule_specificity_limit,None)

    #do_correct_set_subsumption
    def testDoSubInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(do_correct_set_subsumption=2)
        self.assertTrue("do_correct_set_subsumption param must be boolean" in str(context.exception))

    def testDoSub(self):
        clf = ExSTraCS(do_correct_set_subsumption=True)
        self.assertEqual(clf.do_correct_set_subsumption,True)

    #do_GA_subsumption
    def testDoSub2Invalid(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(do_GA_subsumption=2)
        self.assertTrue("do_GA_subsumption param must be boolean" in str(context.exception))

    def testDoSub2(self):
        clf = ExSTraCS(do_GA_subsumption=True)
        self.assertEqual(clf.do_GA_subsumption,True)

    #selection_method
    def testSelectionInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(selection_method="hello")
        self.assertTrue("selection_method param must be 'tournament' or 'roulette'" in str(context.exception))

    def testSelection1(self):
        clf = ExSTraCS(selection_method="tournament")
        self.assertEqual(clf.selection_method,"tournament")

    def testSelection2(self):
        clf = ExSTraCS(selection_method="roulette")
        self.assertEqual(clf.selection_method,"roulette")

    #do_attribute_tracking
    def testDoATInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(do_attribute_tracking=2)
        self.assertTrue("do_attribute_tracking param must be boolean" in str(context.exception))

    def testDoAT(self):
        clf = ExSTraCS(do_attribute_tracking=True)
        self.assertEqual(clf.do_attribute_tracking,True)

    #do_attribute_feedback
    def testDoAFInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(do_attribute_feedback=2)
        self.assertTrue("do_attribute_feedback param must be boolean" in str(context.exception))

    def testDoAF(self):
        clf = ExSTraCS(do_attribute_feedback=True)
        self.assertEqual(clf.do_attribute_feedback,True)

    #expert_knowledge
    def testEKInvalid(self):
        with self.assertRaises(Exception) as context:
            cl = ExSTraCS(expert_knowledge=1)
        self.assertTrue("expert_knowledge param must be None or list/ndarray" in str(context.exception))

    def testEKInvalid2(self):
        with self.assertRaises(Exception) as context:
            cl = ExSTraCS(expert_knowledge="hello")
        self.assertTrue("expert_knowledge param must be None or list/ndarray" in str(context.exception))

    def testEK(self):
        cl = ExSTraCS(expert_knowledge=[1,2,3,4])
        self.assertEqual(cl.expert_knowledge,[1,2,3,4])

    def testEK2(self):
        cl = ExSTraCS(expert_knowledge=np.array([1,2,3,4]))
        self.assertTrue(np.array_equal(cl.expert_knowledge,np.array([1,2,3,4])))

    #rule_compaction
    def testRCInv(self):
        with self.assertRaises(Exception) as context:
            cl = ExSTraCS(rule_compaction="hello")
        self.assertTrue("rule_compaction param must be None or 'QRF' or 'PDRC' or 'QRC' or 'CRA2' or 'Fu2' or 'Fu1'" in str(context.exception))

    def testRCInv2(self):
        with self.assertRaises(Exception) as context:
            cl = ExSTraCS(rule_compaction=2)
        self.assertTrue("rule_compaction param must be None or 'QRF' or 'PDRC' or 'QRC' or 'CRA2' or 'Fu2' or 'Fu1'" in str(context.exception))

    def testRC(self):
        cl = ExSTraCS(rule_compaction=None)
        self.assertEqual(cl.rule_compaction,None)

    def testRC2(self):
        cl = ExSTraCS(rule_compaction='QRF')
        self.assertEqual(cl.rule_compaction,'QRF')

    #reboot_filename
    def testRebootFilenameInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(reboot_filename=2)
        self.assertTrue("reboot_filename param must be None or String from pickle" in str(context.exception))

    def testRebootFilenameInv2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(reboot_filename=True)
        self.assertTrue("reboot_filename param must be None or String from pickle" in str(context.exception))

    def testRebootFilename1(self):
        clf = ExSTraCS()
        self.assertEqual(clf.reboot_filename, None)

    def testRebootFilename2(self):
        clf = ExSTraCS(reboot_filename=None)
        self.assertEqual(clf.reboot_filename, None)

    #discrete_attribute_limit
    def testDiscreteAttributeLimitInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(discrete_attribute_limit="h")
        self.assertTrue("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimitInv2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(discrete_attribute_limit=-10)
        self.assertTrue("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimitInv3(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(discrete_attribute_limit=1.2)
        self.assertTrue("discrete_attribute_limit param must be nonnegative integer or 'c' or 'd'" in str(context.exception))

    def testDiscreteAttributeLimit1(self):
        clf = ExSTraCS(discrete_attribute_limit=10)
        self.assertEqual(clf.discrete_attribute_limit,10)

    def testDiscreteAttributeLimit2(self):
        clf = ExSTraCS(discrete_attribute_limit="c")
        self.assertEqual(clf.discrete_attribute_limit,"c")

    def testDiscreteAttributeLimit3(self):
        clf = ExSTraCS(discrete_attribute_limit="d")
        self.assertEqual(clf.discrete_attribute_limit,"d")

    #specified_attributes
    def testParamSpecAttrNonarray(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(specified_attributes=2)
        self.assertTrue("specified_attributes param must be ndarray" in str(context.exception))

    def testParamSpecAttrNonnumeric(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(specified_attributes=np.array([2,100,"hi",200]))
        self.assertTrue("All specified_attributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttrInvalidNumeric(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(specified_attributes=np.array([2,100,200.2,200]))
        self.assertTrue("All specified_attributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttrInvalidNumeric2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(specified_attributes=np.array([2,100,-200,200]))
        self.assertTrue("All specified_attributes elements param must be nonnegative integers" in str(context.exception))

    def testParamSpecAttr(self):
        clf = ExSTraCS(specified_attributes=np.array([2, 100, 200, 300]))
        self.assertTrue(np.array_equal(clf.specified_attributes,np.array([2, 100, 200, 300])))

    #track_accuracy_while_fit
    def testTrackAccuracyWhileFitInvalid(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(track_accuracy_while_fit=2)
        self.assertTrue("track_accuracy_while_fit param must be boolean" in str(context.exception))

    def testTrackAccuracyWhileFit(self):
        clf = ExSTraCS(track_accuracy_while_fit=True)
        self.assertEqual(clf.track_accuracy_while_fit,True)

    #random_state
    def testRandomSeedInv1(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(random_state="hello")
        self.assertTrue("random_state param must be integer or None" in str(context.exception))

    def testRandomSeedInv2(self):
        with self.assertRaises(Exception) as context:
            clf = ExSTraCS(random_state=1.2)
        self.assertTrue("random_state param must be integer or None" in str(context.exception))

    def testRandomSeed2(self):
        clf = ExSTraCS(random_state=200)
        self.assertEqual(clf.random_state,200)

    def testRandomSeed3(self):
        clf = ExSTraCS(random_state=None)
        self.assertEqual(clf.random_state,None)

    #Performance Tests
    #20B MP Training Accuracy
    def testMultiplexer(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/Multiplexer20Modified.csv")
        converter = StringEnumerator(dataPath,'Class')
        headers, classLabel, dataFeatures, dataPhenotypes = converter.get_params()
        relieffScores = [0.080835, 0.071416, 0.076315, 0.074602, 0.000877, -0.000606, 0.003651, -0.002214, -0.000608,
                  -0.002425, 0.000013, 0.00343, -0.001186, -0.001607, 0.000061, -0.000367, 0.001698, 0.000787, 0.001014,
                  0.001723]
        model = ExSTraCS(learning_iterations=10000, expert_knowledge=relieffScores, N=2000, nu=10)
        model.fit(dataFeatures, dataPhenotypes)
        self.assertTrue(self.approxEqualOrBetter(0.07,model.score(dataFeatures,dataPhenotypes),1.0,True))

    #Continuous Attributes and Missing Training Accuracy
    def testContinuousAndMissing(self):
        dataPath = os.path.join(THIS_DIR, "test/DataSets/Real/ContAndMissing.csv")
        converter = StringEnumerator(dataPath, 'panc_type01')
        converter.delete_attribute("plco_id")
        headers, classLabel, dataFeatures, dataPhenotypes = converter.get_params()
        model = ExSTraCS(learning_iterations=10000,rule_compaction=None)
        model.fit(dataFeatures, dataPhenotypes)
        score = model.score(dataFeatures,dataPhenotypes)
        self.assertTrue(self.approxEqualOrBetter(0.1, score, 0.7, True))

    def approxEqual(self,threshold,comp,right): #threshold is % tolerance
        return abs(abs(comp-right)/right) < threshold

    def approxEqualOrBetter(self,threshold,comp,right,better): #better is False when better is less, True when better is greater
        if not better:
            if self.approxEqual(threshold,comp,right) or comp < right:
                return True
            return False
        else:
            if self.approxEqual(threshold,comp,right) or comp > right:
                return True
            return False


