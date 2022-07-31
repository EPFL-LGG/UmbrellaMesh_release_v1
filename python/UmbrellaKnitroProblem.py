from knitro.numpy import *
import numpy as np

import os
dirname = os.path.dirname(__file__)

class OptKnitroProblem():
    def __init__(self, umbrella_optimizer, update_viewer, minRestLen):
        self.umbrella_optimizer = umbrella_optimizer
        self.update_viewer = update_viewer
        self.minRestLen = minRestLen

    def callbackEvalFCGA (self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFCGA:
            print ("*** callbackEvalFCGA incorrectly called with eval type %d" % evalRequest.type)
            return -1

        evalResult.obj = self.umbrella_optimizer.J(evalRequest.x)
        evalResult.objGrad = self.umbrella_optimizer.gradp_J(evalRequest.x)
        return 0

    def callbackEvalF (self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
            return -1

        evalResult.obj = self.umbrella_optimizer.J(evalRequest.x)
        return 0

    def callbackEvalG (self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
            return -1

        evalResult.objGrad = self.umbrella_optimizer.gradp_J(evalRequest.x)
        return 0

    def callbackEvalHV (self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALHV and evalRequest.type != KN_RC_EVALHV_NO_F:
            print ("*** callbackEvalHV incorrectly called with eval type %d" % evalRequest.type)
            return -1
        
        if (evalRequest.sigma == 0):
            raise Exception("Knitro requested empty Hessian!");

        evalResult.hessVec = self.umbrella_optimizer.apply_hess(evalRequest.x, evalRequest.vec, evalRequest.sigma)
        return 0

    def newPtCallback(self, kc, x, lbda, userParams):
        self.umbrella_optimizer.newPt(x)
        self.update_viewer()
        return 0

    def configureSolver(self, kc, num_steps):
        KN_load_param_file(kc, os.path.join(dirname, "knitro_options.opt"))

        KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW)
        KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE) # 0 minimize, 1 maximize

        KN_set_int_param(kc, KN_PARAM_HONORBNDS, KN_HONORBNDS_ALWAYS); 
        KN_set_int_param(kc, KN_PARAM_MAXIT, num_steps);
        KN_set_int_param(kc, KN_PARAM_PRESOLVE, KN_PRESOLVE_NONE);
        # Set in option file.
        # KN_set_int_param(kc, KN_PARAM_DELTA, trust_region_scale);
        
        KN_set_int_param(kc, KN_PARAM_PAR_NUMTHREADS, 12);
        KN_set_int_param(kc, KN_PARAM_HESSIAN_NO_F, KN_HESSIAN_NO_F_ALLOW);
        KN_set_int_param(kc, KN_PARAM_ALGORITHM, KN_ALG_ACT_CG);
        KN_set_int_param(kc, KN_PARAM_ACT_QPALG, KN_ACT_QPALG_ACT_CG); 
        # Set in option file.
        # KN_set_int_param(kc, KN_PARAM_OPTTOL, optimality_tol);
        KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_ALL);

    def optimize(self, num_steps):
        try:
            kc = KN_new()
        except:
            print("Failed to find a valid license.")
            return -1

        self.configureSolver(kc, num_steps)

        numParams = self.umbrella_optimizer.numParams()
        KN_add_vars(kc, numParams)

        if (self.minRestLen < 0):
            self.minRestLen = self.umbrella_optimizer.defaultLengthBound();

        KN_set_var_lobnds(kc, indexVars=list(np.arange(numParams)), xLoBnds=[self.minRestLen] * numParams)
        KN_set_var_upbnds(kc, indexVars=list(np.arange(numParams)), xUpBnds=[KN_INFINITY] * numParams)

        KN_set_var_primal_init_values(kc, xInitVals=self.umbrella_optimizer.params())

        cb = KN_add_eval_callback(kc, evalObj=None, indexCons=None, funcCallback=self.callbackEvalFCGA)
        KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE, jacIndexVars=KN_DENSE_ROWMAJOR, gradCallback=self.callbackEvalG)
        KN_set_cb_hess(kc, cb, hessIndexVars1=KN_DENSE_ROWMAJOR, hessCallback=self.callbackEvalHV)

        KN_set_newpt_callback(kc, self.newPtCallback, userParams=None)

        try:
            KN_solve(kc)
            solveStatus, objSol, optDoF, lambda_ = KN_get_solution(kc)
            if solveStatus != 0:
                print("KNITRO failed to solve the problem, final status = {}".format(solveStatus))
        except:
            KN_set_newpt_callback(kc, None)
            print("Knitro Interface raised an exception!")
            return -1

        KN_free(kc)

        return solveStatus