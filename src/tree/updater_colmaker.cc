/*!
 * Copyright 2014 by Contributors
 * \file updater_colmaker.cc
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */

  
/* TODO: here are the feature that should be add in this cpp file 
 * -----1.-count and print the task split node num in each tree
 * -----3.-add a weighted calculation of task split_gain_all, otherwise the gain will be negative even if the task is empty at that node.
 * -----4.-add a function to do task_gain_self split
 * -----6.-add a mechanism that if the right tasks all have the positive task gain, we set the will not perform task split at that node.
 * -----10.-merge the param used in this file into the config, then we don't need to build the whole project every time.
 * -----7.-test the result in test data, and make prediction on every center.
 * -----11.-seperate the when and how function, first generate the task split node, then generate the best left tasks, and best right tasks
 * -----9.-add some criteria to make task split in a normalized and non-sensitive way. from Yanru Qu
 * -----2.-add the function of 1-4 split comparing
 * -----13.-compare the loss, but not the AUC. in the result
 * -----5.-add a fucntion to do w* split with 1-4 task split comparing
 * -----15.-set bash param for py training
 * -----12.-tune the parameter on validation set, instead of test set.
 * -----8. -use the positive ratio as a split metric, baseline model
 * -----16.-the OLF loss might be wrong, review it
 * -----18. check whether IsTaskNode() works or not
 * -----25. for OLF split, add a random version that sort the tasks randomly
 * -----17. add some more when and how split functions
 * -----19. make sure the task feature are added each time
 * -----20. output the results with some fixed beginning
 * -----23. count the tasks in the leaf, to see what tasks are usually seperated together
 * -----26. make a random versioin baseline. 
 * -----24. combine the baseline and our method together.
 * -----28. train on the full data.
 * -----35. when split: 5%, 10%, ..., of the gain. 
 * -----30. count the task number in each node at each level. compare with normal xgboost.
 * -----37. check the task split trees, check the results
 * -----33. make some more metrics to see the behaviour of the whole tree.
 * -----44. compute the task gain of each task for each tree, and compare it with the normal xgboost. 
 * 31. compare OLF of the task split and feature split. out put the results. 
 * 21. check the difference between OFL gain and feature gain
 * 22. consider the last value in OLF split, we should omit the tasks with the same task value when propose a possible task split point
 * 27. consider funciton preserving, like make some reweighting for the task.
 * 14. communicate with guoxin and talk about the kdd baselien model
 * 29. try RL, try RL
 * 32. significance test
 * 34. use OLF for feature split top k candidates
 * 36. consider the depth when make the task split 
 * 38. run those tasks together.
 * 39. what about missing data issue. 
 * 40. tune the params when the depth is 4 
 * try MultiTaskElasticNet and other baseline model
 * we can try other dataset.
 * 41. try LR
 * 42. find other mutli task code
 * 43. start working again
 * 45. prohibit task split before the last layer.
 * 
 * 46. count the situations that a child of a task split becomes a non-last layer leaf. -> to support 45
 * 47. calculate the task_gain_all in the task split, this is related to the decidion making in task split
 * 48. figure out how to make the task split decision with the task split task gain in 47  
 * 49. count the ratio of negative task gain 
 * 50. git version control 
 * 
 */




 /*
  FIXME: here are some problems tobe solved
 * -----3. the calculation of task gain all is not stable, fix it.
 * -----2. the w is calculated correctly, needs fix! FIXME:
 * 1. should we calculate task spilt gain in with many other param? like the beta and gamma...
 * 
 *   
 */


#include <xgboost/tree_updater.h>
#include <memory>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include "./param.h"
#include "../common/random.h"
#include "../common/bitmap.h"
#include "../common/sync.h"
#include <iostream>
#include <random>
#include <fstream>  
#include <string>  

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_colmaker);

/*! \brief column-wise update to construct a tree */
template<typename TStats, typename TConstraint>
class ColMaker: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }

  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    TStats::CheckInfo(dmat->Info());
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    TConstraint::Init(&param_, dmat->Info().num_col_);    
    // build tree
    for (auto tree : trees) {
      Builder builder(param_);
      builder.Update(gpair->HostVector(), dmat, tree);
    }
    param_.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param_;
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    TStats stats;
    // the stats of one node contains the following:
      // /*! \brief sum gradient statistics */
      // double sum_grad;
      // /*! \brief sum hessian statistics */
      // double sum_hess;


    /*! \brief extra statistics of data */
    TStats stats_extra;
    /*! \brief last feature value scanned */
    bst_float last_fvalue;
    /*! \brief first feature value scanned */
    bst_float first_fvalue;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit ThreadEntry(const TrainParam &param)
        : stats(param), stats_extra(param) {
    }
    explicit ThreadEntry(const TrainParam &param, const float root_gain)
        : stats(param), stats_extra(param) {
          best.loss_chg=root_gain;
    }
    

  };
  struct NodeEntry {
    /*! \brief statics for node entry */
    TStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain;
    /*! \brief weight calculated related to current data */
    bst_float weight;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit NodeEntry(const TrainParam& param)
        : stats(param), root_gain(0.0f), weight(0.0f){
    }
    explicit NodeEntry(const TrainParam& param, const float root_gain)
        : stats(param), root_gain(root_gain), weight(0.0f){
          best.loss_chg=root_gain;
    }
    
  };
  // actual builder that runs the algorithm
  class Builder {
   public:
    // constructor
    explicit Builder(const TrainParam& param) : param_(param), nthread_(omp_get_max_threads()) {
      tasks_list_.clear();
      for (int task_id: param_.tasks_list_){
        this->tasks_list_.push_back(task_id);
      }
      //this->task_num_for_init_vec=param_.task_num_for_init_vec;    
      this->task_num_for_init_vec=param_.num_task;    
      this->task_num_for_OLF=param_.task_num_for_OLF;
    }
    // update one tree, growing
    virtual void Update(const std::vector<GradientPair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {


      if (param_.xgboost_task_gain_output_flag>0){
        std::ofstream out(param_.output_file,std::ios::app);
        if (out.is_open()){
          out<<"\n";
        }
        out.close();
      }

      accumulate_task_gain_.resize(task_num_for_init_vec,0);


      if (param_.leaf_output_flag>0){
        std::ofstream out(param_.output_file,std::ios::app);
        if (out.is_open()){
          out<<"\n=============================\n\n";
        }
        out.close();  
      }
      srand(param_.baseline_seed);
      this->InitData(gpair, *p_fmat, *p_tree);
      this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);

      // this is the main process of the updating
      for (int depth = 0; depth < param_.max_depth; ++depth) {
      // at print to indicate the layer info
      if (param_.leaf_output_flag>0){
        std::ofstream out(param_.output_file,std::ios::app);
        if (out.is_open()){
          out<<"depth,"<<depth<<",\n";
        }
        out.close();
      }



        this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, &qexpand_);
        this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // print the last depth
      if (param_.leaf_output_flag>0){
        std::ofstream out(param_.output_file,std::ios::app);
        if (out.is_open()){
          out<<"depth,"<<param_.max_depth<<",\n";
        }
        out.close();
      }



      // set all the rest expanding nodes to leaf
      for (size_t i = 0; i < qexpand_.size(); ++i) {
        const int nid = qexpand_[i];
        (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
        if (param_.leaf_output_flag>0){
          // AssignTaskAndNid(RegTree *tree,
          //                       DMatrix *p_fmat,
          //                       std::vector<bst_uint> feat_set,
          //                       const std::vector<int> &qexpand,
          //                       const int num_node_); //TODO: 

          int all_num=0;
          std::ofstream out(param_.output_file,std::ios::app);
          auto num_row=p_fmat->Info().num_row_;
          std::vector<int> task_sample;
          task_sample.resize(task_num_for_init_vec,0);


          for (uint64_t ridx=0; ridx <num_row;ridx++){
            if (DecodePosition(ridx)==nid){
              int task_id = inst_task_id_.at(ridx);
              task_sample.at(task_id)+=1;
              all_num+=1;
            }
          }
          
          if (out.is_open()){
            out<<"l,"<<nid<<","<<snode_[nid].weight * param_.learning_rate<<","<<all_num<<",";
            if (param_.leaf_output_flag==2){
              for (int task_id :tasks_list_){
                out<<task_id<<","<<task_sample.at(task_id)<<",";
              }
            }
          }
          out<<"tasks,";
          int sum_pos_nodes=0;
          for (int task_id:tasks_list_){
            if (task_sample.at(task_id)>0){
              out<<task_id<<",";
              sum_pos_nodes++;
            }
          }
          out<<"sum,"<<sum_pos_nodes<<",";
          out<<"\n";
          out.close();
        }
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
        p_tree->Stat(nid).base_weight = snode_[nid].weight;
        p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
        snode_[nid].stats.SetLeafVec(param_, p_tree->Leafvec(nid));
      }
      // print the tree split node in the tree
      int num_task_node_=0;
      for (bool is_task_node_flag : is_task_node_){
        if (is_task_node_flag){
          num_task_node_+=1;
        }
      }
      LOG(INFO) <<"  "<<num_task_node_<<"  task split nodes, "<<task_node_pruned_num_<< " pruned task nodes";
        
        // this is usded to output the task gain of the whole tree.
        if (param_.task_gain_output_flag>0){
          float temp_sum=0;
          std::ofstream out(param_.output_file,std::ios::app);
          if (out.is_open()){
            for(int task_id:tasks_list_){
              out<<accumulate_task_gain_.at(task_id)<<',';
              temp_sum+=accumulate_task_gain_.at(task_id);
            }
          }
          out<<temp_sum<<",";
          out<<"\n";
          out.close();
        }
      
    }

   protected:
    // initialize temp data structure
    inline void InitData(const std::vector<GradientPair>& gpair,
                         const DMatrix& fmat,
                         const RegTree& tree) {
      CHECK_EQ(tree.param.num_nodes, tree.param.num_roots)
          << "ColMaker: can only grow new tree";
      const std::vector<unsigned>& root_index = fmat.Info().root_index_;
      const RowSet& rowset = fmat.BufferedRowset();
      {
        // setup position
        position_.resize(gpair.size());
        if (root_index.size() == 0) {
          for (size_t i = 0; i < rowset.Size(); ++i) {
            position_[rowset[i]] = 0;
          }
        } else {
          for (size_t i = 0; i < rowset.Size(); ++i) {
            const bst_uint ridx = rowset[i];
            position_[ridx] = root_index[ridx];
            CHECK_LT(root_index[ridx], (unsigned)tree.param.num_roots);
          }
        }
        // mark delete for the deleted datas
        for (size_t i = 0; i < rowset.Size(); ++i) {
          const bst_uint ridx = rowset[i];
          if (gpair[ridx].GetHess() < 0.0f) position_[ridx] = ~position_[ridx];
        }
        // mark subsample
        if (param_.subsample < 1.0f) {
          std::bernoulli_distribution coin_flip(param_.subsample);
          auto& rnd = common::GlobalRandom();
          for (size_t i = 0; i < rowset.Size(); ++i) {
            const bst_uint ridx = rowset[i];
            if (gpair[ridx].GetHess() < 0.0f) continue;
            if (!coin_flip(rnd)) position_[ridx] = ~position_[ridx];
          }
        }
      }
      {
        // initialize feature index
        auto ncol = static_cast<unsigned>(fmat.Info().num_col_);
        for (unsigned i = 0; i < ncol; ++i) {
          if (fmat.GetColSize(i) != 0) {
            feat_index_.push_back(i);
          }
        }
        unsigned n = std::max(static_cast<unsigned>(1),
                              static_cast<unsigned>(param_.colsample_bytree * feat_index_.size()));
        std::shuffle(feat_index_.begin(), feat_index_.end(), common::GlobalRandom());
        CHECK_GT(param_.colsample_bytree, 0U)
            << "colsample_bytree cannot be zero.";
        feat_index_.resize(n);

        // add the task feature if if is dropped by colsample_bytree
        std::vector<bst_uint>::iterator ret = std::find(feat_index_.begin(), feat_index_.end(), 0);
        if (ret==feat_index_.end()){
          feat_index_.push_back(0);
        }
      }
      {
        // setup temp space for each thread
        // reserve a small space
        stemp_.clear();
        stemp_.resize(this->nthread_, std::vector<ThreadEntry>());
        for (size_t i = 0; i < stemp_.size(); ++i) {
          stemp_[i].clear(); stemp_[i].reserve(256);
        }
        snode_.reserve(256);
      }
      {
        // expand query
        qexpand_.reserve(256); qexpand_.clear();
        for (int i = 0; i < tree.param.num_roots; ++i) {
          qexpand_.push_back(i);
        }
      }
    }
    /*!
     * \brief initialize the base_weight, root_gain,
     *  and NodeEntry for all the new nodes in qexpand
     */
    inline void InitNewNode(const std::vector<int>& qexpand,
                            const std::vector<GradientPair>& gpair,
                            const DMatrix& fmat,
                            const RegTree& tree) {
      {
        // setup statistics space for each tree node
        for (size_t i = 0; i < stemp_.size(); ++i) {
          stemp_[i].resize(tree.param.num_nodes, ThreadEntry(param_));
        }
        snode_.resize(tree.param.num_nodes, NodeEntry(param_));
        constraints_.resize(tree.param.num_nodes);
      }
      const RowSet &rowset = fmat.BufferedRowset();
      const MetaInfo& info = fmat.Info();
      // setup position
      const auto ndata = static_cast<bst_omp_uint>(rowset.Size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int tid = omp_get_thread_num();
        if (position_[ridx] < 0) continue;
        stemp_[tid][position_[ridx]].stats.Add(gpair, info, ridx);
      }
      // sum the per thread statistics together
      for (int nid : qexpand) {
        TStats stats(param_);
        for (size_t tid = 0; tid < stemp_.size(); ++tid) {
          stats.Add(stemp_[tid][nid].stats);
        }
        // update node statistics
        snode_[nid].stats = stats;
      }
      // setup constraints before calculating the weight
      for (int nid : qexpand) {
        if (tree[nid].IsRoot()) continue;
        const int pid = tree[nid].Parent();
        constraints_[pid].SetChild(param_, tree[pid].SplitIndex(),
                                   snode_[tree[pid].LeftChild()].stats,
                                   snode_[tree[pid].RightChild()].stats,
                                   &constraints_[tree[pid].LeftChild()],
                                   &constraints_[tree[pid].RightChild()]);
      }
      // calculating the weights
      for (int nid : qexpand) {
        snode_[nid].root_gain = static_cast<float>(
            constraints_[nid].CalcGain(param_, snode_[nid].stats));
        snode_[nid].weight = static_cast<float>(
            constraints_[nid].CalcWeight(param_, snode_[nid].stats));
      }
    }
    /*! \brief update queue expand add in new leaves */
    inline void UpdateQueueExpand(const RegTree& tree, std::vector<int>* p_qexpand) {
      std::vector<int> &qexpand = *p_qexpand;
      std::vector<int> newnodes;
      for (int nid : qexpand) {
        if (!tree[ nid ].IsLeaf()) {
          newnodes.push_back(tree[nid].LeftChild());
          newnodes.push_back(tree[nid].RightChild());
        }
      }
      // use new nodes for qexpand
      qexpand = newnodes;
    }
    // parallel find the best split of current fid
    // this function does not support nested functions
    inline void ParallelFindSplit(const ColBatch::Inst &col,
                                  bst_uint fid,
                                  const DMatrix &fmat,
                                  const std::vector<GradientPair> &gpair) {
      // TODO(tqchen): double check stats order.
      const MetaInfo& info = fmat.Info();
      const bool ind = col.length != 0 && col.data[0].fvalue == col.data[col.length - 1].fvalue;
      bool need_forward = param_.NeedForwardSearch(fmat.GetColDensity(fid), ind);
      bool need_backward = param_.NeedBackwardSearch(fmat.GetColDensity(fid), ind);
      const std::vector<int> &qexpand = qexpand_;
      #pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        std::vector<ThreadEntry> &temp = stemp_[tid];
        // cleanup temp statistics
        for (int j : qexpand) {
          temp[j].stats.Clear();
        }
        bst_uint step = (col.length + this->nthread_ - 1) / this->nthread_;
        bst_uint end = std::min(col.length, step * (tid + 1));
        for (bst_uint i = tid * step; i < end; ++i) {
          const bst_uint ridx = col[i].index;
          const int nid = position_[ridx];
          if (nid < 0) continue;
          const bst_float fvalue = col[i].fvalue;
          if (temp[nid].stats.Empty()) {
            temp[nid].first_fvalue = fvalue;
          }
          temp[nid].stats.Add(gpair, info, ridx);
          temp[nid].last_fvalue = fvalue;
        }
      }
      // start collecting the partial sum statistics
      auto nnode = static_cast<bst_omp_uint>(qexpand.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint j = 0; j < nnode; ++j) {
        const int nid = qexpand[j];
        TStats sum(param_), tmp(param_), c(param_);
        for (int tid = 0; tid < this->nthread_; ++tid) {
          tmp = stemp_[tid][nid].stats;
          stemp_[tid][nid].stats = sum;
          sum.Add(tmp);
          if (tid != 0) {
            std::swap(stemp_[tid - 1][nid].last_fvalue, stemp_[tid][nid].first_fvalue);
          }
        }
        for (int tid = 0; tid < this->nthread_; ++tid) {
          stemp_[tid][nid].stats_extra = sum;
          ThreadEntry &e = stemp_[tid][nid];
          bst_float fsplit;
          if (tid != 0) {
            if (stemp_[tid - 1][nid].last_fvalue != e.first_fvalue) {
              fsplit = (stemp_[tid - 1][nid].last_fvalue + e.first_fvalue) * 0.5f;
            } else {
              continue;
            }
          } else {
            fsplit = e.first_fvalue - kRtEps;
          }
          if (need_forward && tid != 0) {
            c.SetSubstract(snode_[nid].stats, e.stats);
            if (c.sum_hess >= param_.min_child_weight &&
                e.stats.sum_hess >= param_.min_child_weight) {
              auto loss_chg = static_cast<bst_float>(
                  constraints_[nid].CalcSplitGain(
                      param_, param_.monotone_constraints[fid], e.stats, c) -
                  snode_[nid].root_gain);
              e.best.Update(loss_chg, fid, fsplit, false);
            }
          }
          if (need_backward) {
            tmp.SetSubstract(sum, e.stats);
            c.SetSubstract(snode_[nid].stats, tmp);
            if (c.sum_hess >= param_.min_child_weight &&
                tmp.sum_hess >= param_.min_child_weight) {
              auto loss_chg = static_cast<bst_float>(
                  constraints_[nid].CalcSplitGain(
                      param_, param_.monotone_constraints[fid], tmp, c) -
                  snode_[nid].root_gain);
              e.best.Update(loss_chg, fid, fsplit, true);
            }
          }
        }
        if (need_backward) {
          tmp = sum;
          ThreadEntry &e = stemp_[this->nthread_-1][nid];
          c.SetSubstract(snode_[nid].stats, tmp);
          if (c.sum_hess >= param_.min_child_weight &&
              tmp.sum_hess >= param_.min_child_weight) {
            auto loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(
                    param_, param_.monotone_constraints[fid], tmp, c) -
                snode_[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + kRtEps, true);
          }
        }
      }
      // rescan, generate candidate split
      #pragma omp parallel
      {
        TStats c(param_), cright(param_);
        const int tid = omp_get_thread_num();
        std::vector<ThreadEntry> &temp = stemp_[tid];
        bst_uint step = (col.length + this->nthread_ - 1) / this->nthread_;
        bst_uint end = std::min(col.length, step * (tid + 1));
        for (bst_uint i = tid * step; i < end; ++i) {
          const bst_uint ridx = col[i].index;
          const int nid = position_[ridx];
          if (nid < 0) continue;
          const bst_float fvalue = col[i].fvalue;
          // get the statistics of nid
          ThreadEntry &e = temp[nid];
          if (e.stats.Empty()) {
            e.stats.Add(gpair, info, ridx);
            e.first_fvalue = fvalue;
          } else {
            // forward default right
            if (fvalue != e.first_fvalue) {
              if (need_forward) {
                c.SetSubstract(snode_[nid].stats, e.stats);
                if (c.sum_hess >= param_.min_child_weight &&
                    e.stats.sum_hess >= param_.min_child_weight) {
                  auto loss_chg = static_cast<bst_float>(
                      constraints_[nid].CalcSplitGain(
                          param_, param_.monotone_constraints[fid], e.stats, c) -
                      snode_[nid].root_gain);
                  e.best.Update(loss_chg, fid, (fvalue + e.first_fvalue) * 0.5f,
                                false);
                }
              }
              if (need_backward) {
                cright.SetSubstract(e.stats_extra, e.stats);
                c.SetSubstract(snode_[nid].stats, cright);
                if (c.sum_hess >= param_.min_child_weight &&
                    cright.sum_hess >= param_.min_child_weight) {
                  auto loss_chg = static_cast<bst_float>(
                      constraints_[nid].CalcSplitGain(
                          param_, param_.monotone_constraints[fid], c, cright) -
                      snode_[nid].root_gain);
                  e.best.Update(loss_chg, fid, (fvalue + e.first_fvalue) * 0.5f, true);
                }
              }
            }
            e.stats.Add(gpair, info, ridx);
            e.first_fvalue = fvalue;
          }
        }
      }
    }
    // update enumeration solution
    inline void UpdateEnumeration(int nid, GradientPair gstats,
                                  bst_float fvalue, int d_step, bst_uint fid,
                                  TStats &c, std::vector<ThreadEntry> &temp) { // NOLINT(*)
      // get the statistics of nid
      ThreadEntry &e = temp[nid];
      // test if first hit, this is fine, because we set 0 during init
      if (e.stats.Empty()) {
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      } else {
        // try to find a split
        if (fvalue != e.last_fvalue &&
            e.stats.sum_hess >= param_.min_child_weight) {
          c.SetSubstract(snode_[nid].stats, e.stats);
  // /*! \brief set current value to a - b */
  // inline void SetSubstract(const GradStats& a, const GradStats& b) {
  //   sum_grad = a.sum_grad - b.sum_grad;
  //   sum_hess = a.sum_hess - b.sum_hess;
  // }
  // in a stats
          if (c.sum_hess >= param_.min_child_weight) {
            bst_float loss_chg;
            if (d_step == -1) {
              loss_chg = static_cast<bst_float>(
                  constraints_[nid].CalcSplitGain(
                      param_, param_.monotone_constraints[fid], c, e.stats) -
                  snode_[nid].root_gain);
            } else {
              loss_chg = static_cast<bst_float>(
                  constraints_[nid].CalcSplitGain(
                      param_, param_.monotone_constraints[fid], e.stats, c) -
                  snode_[nid].root_gain);
            }
            e.best.Update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f,
                          d_step == -1);
          }
        }
        // update the statistics
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      }
    }
    // same as EnumerateSplit, with cacheline prefetch optimization
    inline void EnumerateSplitCacheOpt(const ColBatch::Entry *begin,
                                       const ColBatch::Entry *end,
                                       int d_step,
                                       bst_uint fid,
                                       const std::vector<GradientPair> &gpair,
                                       std::vector<ThreadEntry> &temp) { // NOLINT(*)
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats.Clear();//FIXME: we must know what is temp, what is ThreadEntry
      }
      // left statistics
      TStats c(param_);
      // local cache buffer for position and gradient pair
      constexpr int kBuffer = 32;
      int buf_position[kBuffer] = {};
      GradientPair buf_gpair[kBuffer] = {};
      // aligned ending position
      const ColBatch::Entry *align_end;
      if (d_step > 0) {
        align_end = begin + (end - begin) / kBuffer * kBuffer;
      } else {
        align_end = begin - (begin - end) / kBuffer * kBuffer;
      }
      int i;
      const ColBatch::Entry *it;
      const int align_step = d_step * kBuffer;
      // internal cached loop
      for (it = begin; it != align_end; it += align_step) {
        const ColBatch::Entry *p;
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          buf_position[i] = position_[p->index];
          buf_gpair[i] = gpair[p->index];
        }
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          const int nid = buf_position[i];
          if (nid < 0) continue;
          this->UpdateEnumeration(nid, buf_gpair[i],
                                  p->fvalue, d_step,
                                  fid, c, temp);
        }
      }
      // finish up the ending piece
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        buf_position[i] = position_[it->index];
        buf_gpair[i] = gpair[it->index];
      }
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        const int nid = buf_position[i];
        if (nid < 0) continue;
        this->UpdateEnumeration(nid, buf_gpair[i],
                                it->fvalue, d_step,
                                fid, c, temp);
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode_[nid].stats, e.stats);
        if (e.stats.sum_hess >= param_.min_child_weight &&
            c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(
                    param_, param_.monotone_constraints[fid], c, e.stats) -
                snode_[nid].root_gain);
          } else {
            loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(
                    param_, param_.monotone_constraints[fid], e.stats, c) -
                snode_[nid].root_gain);
          }
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }               
    }

    // enumerate the split values of specific feature
    inline void EnumerateSplit(const ColBatch::Entry *begin,
                               const ColBatch::Entry *end,
                               int d_step,
                               bst_uint fid,
                               const std::vector<GradientPair> &gpair,
                               const MetaInfo &info,
                               std::vector<ThreadEntry> &temp) { // NOLINT(*)
                               // TODO: this function might be useful, for 1->4 task split compare
      // use cacheline aware optimization
      if (TStats::kSimpleStats != 0 && param_.cache_opt != 0) {
        EnumerateSplitCacheOpt(begin, end, d_step, fid, gpair, temp);
        return;
      }
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats.Clear();  // for a new round computing for one specific feature at each node.
      }
      // left statistics
      TStats c(param_);// now I understand what does left means, it means the rest of the samples' statistics
      for (const ColBatch::Entry *it = begin; it != end; it += d_step) {
        const bst_uint ridx = it->index;
        const int nid = position_[ridx];
        if (nid < 0) continue;
        // start working
        const bst_float fvalue = it->fvalue;
        // get the statistics of nid
        ThreadEntry &e = temp[nid];    
        // test if first hit, this is fine, because we set 0 during init
        if (e.stats.Empty()) { 
          e.stats.Add(gpair, info, ridx);  
          e.last_fvalue = fvalue;   
        } else {
          // try to find a split
          if (fvalue != e.last_fvalue &&  // this condition ensures that the same fvalue comes continuelly will not make any difference.  
              e.stats.sum_hess >= param_.min_child_weight) {
            c.SetSubstract(snode_[nid].stats, e.stats);    // snode_[nid] is the (current) best result, compare to the result in this thread (feature)
            if (c.sum_hess >= param_.min_child_weight) {  // the two '>= param_.min_child_weight' means the two child nodes' sum_hess should meet the constaint 
              bst_float loss_chg;
              if (d_step == -1) {
                loss_chg = static_cast<bst_float>(
                    constraints_[nid].CalcSplitGain( 
                        param_, param_.monotone_constraints[fid], c, e.stats) -
                    snode_[nid].root_gain); 
              } else {
                loss_chg = static_cast<bst_float>(
                    constraints_[nid].CalcSplitGain(
                        param_, param_.monotone_constraints[fid], e.stats, c) -
                    snode_[nid].root_gain);
              }
              e.best.Update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, d_step == -1);
            }
          }
          // update the statistics
          e.stats.Add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        }
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {// TODO: don't understand the purpose here? did't we find the best result? 
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode_[nid].stats, e.stats);
        if (e.stats.sum_hess >= param_.min_child_weight &&
            c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(
                    param_, param_.monotone_constraints[fid], c, e.stats) -
                snode_[nid].root_gain);
          } else {
            loss_chg = static_cast<bst_float>(
                constraints_[nid].CalcSplitGain(
                    param_, param_.monotone_constraints[fid], e.stats, c) -
                snode_[nid].root_gain);
          }
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }
    }



    // enumerate the split values of specific feature
    inline void EnumerateSplitTask(const ColBatch::Entry *begin,
                               const ColBatch::Entry *end,
                               int d_step,
                               bst_uint fid,
                               const std::vector<GradientPair> &gpair,
                               const MetaInfo &info,
                               std::vector<ThreadEntry> &temp,
                               const std::vector<int> &qexpand,
                               const std::vector<int> &position_task,
                               const std::vector<bool> &sub_node_works) { // NOLINT(*)
                               // TODO: this function might be useful, for 1->4 task split compare
      
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats.Clear();  // for a new round computing for one specific feature at each node.
      }
      // left statistics
      TStats c(param_);// now I understand what does left means, it means the rest of the samples' statistics
      for (const ColBatch::Entry *it = begin; it != end; it += d_step) {
        const bst_uint ridx = it->index;
        const int nid = position_task[ridx];
        
        if (nid < 0) continue;
        if (!sub_node_works.at(nid)) continue;
        // start working
        const bst_float fvalue = it->fvalue;

        // get the statistics of nid
        ThreadEntry &e = temp[nid];    
        // test if first hit, this is fine, because we set 0 during init
        if (e.stats.Empty()) { 
          e.stats.Add(gpair, info, ridx);  
          e.last_fvalue = fvalue;   
        } else {
          // try to find a split
          if (fvalue != e.last_fvalue // this condition ensures that the same fvalue comes continuelly will not make any difference.  
              // &&   e.stats.sum_hess >= param_.min_child_weight  //FIXME: need to delete this, otherwise the sesult will be weird
              ) {
            c.SetSubstract(snode_task_[nid].stats, e.stats);    // snode_task_[nid] is the (current) best result, compare to the result in this thread (feature)
            //if (c.sum_hess >= param_.min_child_weight) {  // the two '>= param_.min_child_weight' means the two child nodes' sum_hess should meet the constaint 
              bst_float loss_chg;
              if (d_step == -1) {
                loss_chg = static_cast<bst_float>(
                    constraints_task_[nid].CalcSplitGain( 
                        param_, param_.monotone_constraints[fid], c, e.stats) -
                    snode_task_[nid].root_gain); 
              } else {
                loss_chg = static_cast<bst_float>(
                    constraints_task_[nid].CalcSplitGain(
                        param_, param_.monotone_constraints[fid], e.stats, c) -
                    snode_task_[nid].root_gain);
              }
              e.best.Update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, d_step == -1);
            //}
          }
          // update the statistics
          e.stats.Add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        }
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {// TODO: don't understand the purpose here? did't we find the best result? 
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode_task_[nid].stats, e.stats);
        //if (e.stats.sum_hess >= param_.min_child_weight && c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                constraints_task_[nid].CalcSplitGain(
                    param_, param_.monotone_constraints[fid], c, e.stats) -
                snode_task_[nid].root_gain);
          } else {
            loss_chg = static_cast<bst_float>(
                constraints_task_[nid].CalcSplitGain(
                    param_, param_.monotone_constraints[fid], e.stats, c) -
                snode_task_[nid].root_gain);
          }
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        //}
      }
    }// TODO: double check!

    // update the solution candidate
    virtual void UpdateSolution(const ColBatch& batch,
                                const std::vector<GradientPair>& gpair,
                                const DMatrix& fmat) {
      const MetaInfo& info = fmat.Info();
      // start enumeration
      const auto nsize = static_cast<bst_omp_uint>(batch.size);   // nsize is the number of features, this means feature level parallel
      #if defined(_OPENMP)
      const int batch_size = std::max(static_cast<int>(nsize / this->nthread_ / 32), 1);
      #endif
      int poption = param_.parallel_option;
      if (poption == 2) {
        poption = static_cast<int>(nsize) * 2 < this->nthread_ ? 1 : 0;
      }
      if (poption == 0) {
        #pragma omp parallel for schedule(dynamic, batch_size)
        for (bst_omp_uint i = 0; i < nsize; ++i) {

          // i is the i-th feature
          // c is the i-th feature data (col)

          const bst_uint fid = batch.col_index[i];
          const int tid = omp_get_thread_num();
          const ColBatch::Inst c = batch[i];
          const bool ind = c.length != 0 && c.data[0].fvalue == c.data[c.length - 1].fvalue;
          if (param_.NeedForwardSearch(fmat.GetColDensity(fid), ind)) {
            this->EnumerateSplit(c.data, c.data + c.length, +1,
                                 fid, gpair, info, stemp_[tid]);
          }
          if (param_.NeedBackwardSearch(fmat.GetColDensity(fid), ind)) {
            this->EnumerateSplit(c.data + c.length - 1, c.data - 1, -1,
                                 fid, gpair, info, stemp_[tid]);
          }
        }
      } else {
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          this->ParallelFindSplit(batch[i], batch.col_index[i],
                                  fmat, gpair);
        }
      }
    }




    inline void AssignTaskAndNid(RegTree *tree,
                                DMatrix *p_fmat,
                                std::vector<bst_uint> feat_set,
                                const std::vector<int> &qexpand,
                                const int num_node_){

      std::vector<bst_uint> nid_split_index;
      nid_split_index.clear();
      nid_split_index.resize(num_node_,0);
      std::vector<bst_float> nid_split_cond;
      nid_split_cond.clear();
      nid_split_cond.resize(num_node_,0);

      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        fsplits.push_back(e.best.SplitIndex());
        nid_split_index.at(nid)=e.best.SplitIndex();
        nid_split_cond.at(nid)=e.best.split_value;
      }  


      if (param_.debug==6){
        for (int nid : qexpand){
          std::cout<<nid<<":"<<nid_split_index.at(nid)<<"\t"<<nid_split_cond.at(nid)<<"\n";
        }
      }

      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());

      dmlc::DataIter<ColBatch> *iter_task = p_fmat->ColIterator(fsplits);
      
      // pre-split the data, get the left or right info
      while (iter_task->Next()) {  
        const ColBatch &batch = iter_task->Value();
        if (param_.debug==15){
          std::cout<<batch.size<<"\\\\\\\n";
        }
        for (size_t i = 0; i < batch.size; ++i) {

          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const auto ndata = static_cast<bst_omp_uint>(col.length);
          if (param_.debug==15){
          std::cout<<ndata<<"====="<<fid<<"\n";
          }
          // #pragma omp parallel for schedule(static)   // FIXME: the code here is wrong if we use OMP
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = DecodePosition(ridx);  // HINT: remember this mistake, don't trust the name of data
            inst_nid_.at(ridx)=nid;
            if (param_.debug==13){
              // std::cout<<nid<<"|"<<ridx<<"|"<<ndata<<"\t";
            }
            const bst_float fvalue = col[j].fvalue;
            if (nid_split_index.at(nid) == fid) {
              node_inst_num_.at(nid)+=1; 
              if (fvalue < nid_split_cond.at(nid)) {
                inst_go_left_.at(ridx)=true;
              } else {
                inst_go_left_.at(ridx)=false;
              }
            }
          }
        }
      }
      
      // find the task info, and calculate the task inst num at each node
      iter_task = p_fmat->ColIterator(feat_set);
      while (iter_task->Next()) {
          auto batch = iter_task->Value();
          const auto nsize = static_cast<bst_omp_uint>(batch.size);
          bst_omp_uint task_i;
          for (bst_omp_uint i = 0; i < nsize; ++i) {
          const bst_uint fid = batch.col_index[i];  // should check the feature idx, i is not the idx.
          if (fid==0){
            task_i = i; // find the task idx task_i
            break;
            }
          }  
          const ColBatch::Inst task_c = batch[task_i];
          for (const ColBatch::Entry *it = task_c.data; it != task_c.data+task_c.length; it += 1) {
            const bst_uint ridx = it->index;
            const bst_float fvalue = it->fvalue;
            int task_id=int(fvalue);

            const int nid = DecodePosition(ridx);

            inst_task_id_.at(ridx)=task_id;
          
            if (inst_go_left_.at(ridx)){
              node_task_inst_num_left_.at(nid).at(task_id)+=1;
              node_inst_num_left_.at(nid)+=1;
            }
            else{
              if (param_.debug==2){
              // if (nid==216 || task_id==216)
              std::cout<<nid<<"|"<<task_id<<"\t";
            }
              node_task_inst_num_right_.at(nid).at(task_id)+=1;
              node_inst_num_right_.at(nid)+=1;
            }
            
            
            
          }
        }
    }

    inline void InitAuxiliary(const uint64_t num_row_, const int num_node_,const std::vector<int> &qexpand){
      /****************************** init auxiliary val ***********************************/
      // they can be init at the begining of each depth before the real task split

      // the nid of each inst
      // data_in
      inst_nid_.clear();
      inst_nid_.resize(num_row_,-1);
      // the task_id of each isnt
      inst_task_id_.clear();
      inst_task_id_.resize(num_row_,-1);
      // whether the inst goes left (true) or right (false)
      inst_go_left_.clear();
      inst_go_left_.resize(num_row_,true);


      // G and H in capital means the sum of the G and H over all the instances
      // store the G and H in the node, in order to calculate w* for the whole node 
      // auto biggest = std::max_element(std::begin(qexpand), std::end(qexpand));
      // std::cout<<*biggest<<"\t\n";
      G_node_.resize(num_node_);
      H_node_.resize(num_node_);
      is_task_node_.resize(num_node_);
      for (int nid : qexpand) {
        G_node_.at(nid)=0;
        H_node_.at(nid)=0;
        is_task_node_.at(nid)=false;
      }

    


      // store the G and H for each task in each node's left and right child, the 
      G_task_lnode_.resize(num_node_);
      G_task_rnode_.resize(num_node_);
      H_task_lnode_.resize(num_node_);
      H_task_rnode_.resize(num_node_);

      task_gain_self_.resize(num_node_);
      task_gain_all_.resize(num_node_);
      node_task_inst_num_left_.resize(num_node_);
      node_task_inst_num_right_.resize(num_node_);

      node_pos_ratio_.clear();
      node_pos_ratio_.resize(num_node_,0);

      // store the left and right task task_idx for each task split node 
      task_node_left_tasks_.resize(num_node_);
      task_node_right_tasks_.resize(num_node_);
      
      node_inst_num_left_.resize(num_node_);
      node_inst_num_right_.resize(num_node_);
      node_inst_num_.resize(num_node_);
      node_task_pos_ratio_.resize(num_node_);
      
      // we can resize the tasks when needed. 


      task_w_.resize(num_node_);
      for (int nid : qexpand) {
        G_task_lnode_.at(nid).resize(task_num_for_init_vec,-1);
        G_task_rnode_.at(nid).resize(task_num_for_init_vec,-1);
        H_task_lnode_.at(nid).resize(task_num_for_init_vec,-1);
        H_task_rnode_.at(nid).resize(task_num_for_init_vec,-1);
        task_gain_self_.at(nid).resize(task_num_for_init_vec,-1);
        task_gain_all_.at(nid).resize(task_num_for_init_vec,-1);
        task_w_.at(nid).resize(task_num_for_init_vec,-1);
        node_task_inst_num_left_.at(nid).resize(task_num_for_init_vec,-1);
        node_task_inst_num_right_.at(nid).resize(task_num_for_init_vec,-1);
        node_task_pos_ratio_.at(nid).resize(task_num_for_init_vec,-1);
        for (int task_id : tasks_list_){
        
          G_task_lnode_.at(nid).at(task_id)=0;
          G_task_rnode_.at(nid).at(task_id)=0;
          H_task_lnode_.at(nid).at(task_id)=0;
          H_task_rnode_.at(nid).at(task_id)=0;
          task_gain_self_.at(nid).at(task_id)=0;
          task_gain_all_.at(nid).at(task_id)=0;
          task_w_.at(nid).at(task_id)=0;

          node_task_inst_num_left_.at(nid).at(task_id)=0;
          node_task_inst_num_right_.at(nid).at(task_id)=0;
          node_task_pos_ratio_.at(nid).at(task_id)=0;
        }  
      }

    }

    inline void CalcGHInNode(int num_row,const std::vector<GradientPair> &gpair){
      for (int ridx =0;ridx<num_row;ridx++){
        // get the gradient 
        const GradientPair& b = gpair[ridx];
        int nid=inst_nid_.at(ridx);
        int task_id=inst_task_id_.at(ridx);

        //calcu G_node and H_node
        if (param_.debug==13){
          std::cout<<"!"<<nid<<"|"<<ridx<<"\t";
        }
        G_node_.at(nid)+=b.GetGrad();
        H_node_.at(nid)+=b.GetHess();

        if (inst_go_left_.at(ridx)){
          G_task_lnode_.at(nid).at(task_id)+=b.GetGrad();
          H_task_lnode_.at(nid).at(task_id)+=b.GetHess();
        }  
        else {
          G_task_rnode_.at(nid).at(task_id)+=b.GetGrad();
          H_task_rnode_.at(nid).at(task_id)+=b.GetHess();
        }
      }
    }


    inline float Square(float x){return x*x;}


    

    inline void CalcTaskWStar(const std::vector<int> &qexpand){
      for (int nid : qexpand){
        for (int task_id : tasks_list_){
          task_w_.at(nid).at(task_id)= - ( G_task_rnode_.at(nid).at(task_id) + G_task_lnode_.at(nid).at(task_id)  ) / 
                                       ( H_task_rnode_.at(nid).at(task_id) + H_task_lnode_.at(nid).at(task_id) + param_.reg_lambda);        
        }
      }
    }

    inline void CalcTaskGainSelf(const std::vector<int> &qexpand){
      for (int nid : qexpand){
        //TODO: should we follow the whole setting of xgboost? I mean the beta and gamma term if xgboost's gain calculation.
        for (int task_id : tasks_list_){
          float gain_left= Square(G_task_lnode_.at(nid).at(task_id)) / (H_task_lnode_.at(nid).at(task_id)+param_.reg_lambda);
          float gain_right= Square(G_task_rnode_.at(nid).at(task_id)) / (H_task_rnode_.at(nid).at(task_id)+param_.reg_lambda);
          float gain=Square(G_task_lnode_.at(nid).at(task_id)+G_task_rnode_.at(nid).at(task_id))/
                    (H_task_lnode_.at(nid).at(task_id)+H_task_rnode_.at(nid).at(task_id)+param_.reg_lambda);

          task_gain_self_.at(nid).at(task_id)=0.5*(gain_left+gain_right-gain); //FIXME: should we subtract param_.gamma 
        }
      }
    }
    inline void CalcTaskGainAll(const std::vector<int> &qexpand){
      for (int nid : qexpand){
        // cal w_l and w_r
        float G_L=0, G_R=0, H_L=0, H_R=0;
        for (int task_id : tasks_list_){
          G_L+=G_task_lnode_.at(nid).at(task_id);
          G_R+=G_task_rnode_.at(nid).at(task_id);
          H_L+=H_task_lnode_.at(nid).at(task_id);
          H_R+=H_task_rnode_.at(nid).at(task_id);
        }
        float w_L=-G_L/(H_L+param_.reg_lambda);
        float w_R=-G_R/(H_R+param_.reg_lambda);
        float w=-(G_L+G_R)/(H_L+H_R+param_.reg_lambda);
        // TODO: fix the lambda here!
        for (int task_id : tasks_list_){
          float gain_left = -(G_task_lnode_.at(nid).at(task_id)*w_L+0.5*(H_task_lnode_.at(nid).at(task_id)+param_.reg_lambda)*Square(w_L));
          float gain_right = -(G_task_rnode_.at(nid).at(task_id)*w_R+0.5*(H_task_rnode_.at(nid).at(task_id)+ param_.reg_lambda)*Square(w_R));
          float gain = ( (G_task_lnode_.at(nid).at(task_id) + G_task_rnode_.at(nid).at(task_id) ) * w
                       +0.5*( H_task_lnode_.at(nid).at(task_id) + H_task_rnode_.at(nid).at(task_id) + param_.reg_lambda ) * Square(w));
          task_gain_all_.at(nid).at(task_id) = gain_left+gain_right+gain;
        }
      }
    }

    // inline void AccumulateTaskGain(const std::vector<int> &qexpand){
    //   for (int nid: qexpand){
    //     for (int task_id : tasks_list_){
    //       accumulate_task_gain_.at(task_id)+=task_gain_all_.at(nid).at(task_id);
    //     }
    //   }
    // }

    inline void CalcTaskGainAllLambdaWeighted(const std::vector<int> &qexpand){
      for (int nid : qexpand){
        // cal w_l and w_r
        float G_L=0, G_R=0, H_L=0, H_R=0;
        for (int task_id : tasks_list_){
          G_L+=G_task_lnode_.at(nid).at(task_id);
          G_R+=G_task_rnode_.at(nid).at(task_id);
          H_L+=H_task_lnode_.at(nid).at(task_id);
          H_R+=H_task_rnode_.at(nid).at(task_id);
        }
        float w_L=-G_L/(H_L+param_.reg_lambda);
        float w_R=-G_R/(H_R+param_.reg_lambda);
        float w=-(G_L+G_R)/(H_L+H_R+param_.reg_lambda);
        // TODO: fix the lambda here!
        for (int task_id : tasks_list_){
            float left_lambda = 0;
            float right_lambda = 0;
            float lambda = 0;
            float gamma = 0;
            
          
          if (node_inst_num_left_.at(nid)!=0){
            left_lambda = param_.reg_lambda*(node_task_inst_num_left_.at(nid).at(task_id)/node_inst_num_left_.at(nid));
          }
          else{
            left_lambda = 0;
          }
          if (node_inst_num_right_.at(nid)!=0){
            right_lambda = param_.reg_lambda*(node_task_inst_num_right_.at(nid).at(task_id)/node_inst_num_right_.at(nid));
          }
          else{
            right_lambda = 0;
          }
          if ( ( node_inst_num_right_.at(nid) + node_inst_num_left_.at(nid) ) != 0 ){
            lambda = param_.reg_lambda*( ( node_task_inst_num_right_.at(nid).at(task_id) + node_task_inst_num_left_.at(nid).at(task_id) )
                            / ( node_inst_num_right_.at(nid) + node_inst_num_left_.at(nid) ) );
            gamma = param_.min_split_loss*( ( node_task_inst_num_right_.at(nid).at(task_id) + node_task_inst_num_left_.at(nid).at(task_id) )
                            / ( node_inst_num_right_.at(nid) + node_inst_num_left_.at(nid) ) );
          }
          else{
            lambda = 0;
            gamma = 0;
          }
          float gain_left = -(G_task_lnode_.at(nid).at(task_id)*w_L+0.5*(H_task_lnode_.at(nid).at(task_id)+left_lambda)*Square(w_L));
          float gain_right = -(G_task_rnode_.at(nid).at(task_id)*w_R+0.5*(H_task_rnode_.at(nid).at(task_id)+ right_lambda)*Square(w_R));
          float gain = ( (G_task_lnode_.at(nid).at(task_id) + G_task_rnode_.at(nid).at(task_id) ) * w
                       +0.5*( H_task_lnode_.at(nid).at(task_id) + H_task_rnode_.at(nid).at(task_id) + lambda ) * Square(w));
          task_gain_all_.at(nid).at(task_id) = gain_left+gain_right+gain-gamma;
        }
      }
    }  // FIXME: the calculation of task gain all is not stable, fix it


    inline bool WhenTaskSplitHardMargin(std::vector<std::vector<float> > * task_gain_,int nid, float min_task_gain){
      for (float task_gain : task_gain_->at(nid)){
        if (task_gain<min_task_gain){
          return true;
        }
      }
      return false;
    }
    inline bool WhenTaskSplitRandom(){
      float ran_float = rand()*1.0/(RAND_MAX*1.0);
      
      if (ran_float < param_.baseline_lambda){
        return true;
      }
      else{
        return false;
      }
    }

    // when the sample ratio of tasks that have negative task gain is larger than threshold_ratio_R, we set this node as a task split node
    inline bool WhenTaskSplitNegativeSampleRatio(std::vector<std::vector<float> > * task_gain_,int nid, float threshold_ratio_R){
      
      float neg_task_gain_sample_num=0;
      float sample_num=( node_inst_num_right_.at(nid) + node_inst_num_left_.at(nid) );
      float task_gain =0;
      float gain_all_at_node=0;
      
      for (int task_id : tasks_list_){
        // if (nid == param_.nid_debug){

          task_gain = task_gain_->at(nid).at(task_id);
          gain_all_at_node+=task_gain;


          if (task_gain<0){
            neg_task_gain_sample_num+=( node_task_inst_num_right_.at(nid).at(task_id) + node_task_inst_num_left_.at(nid).at(task_id) );
          }
        if (nid == param_.nid_debug){
          if (param_.debug==1){
            std::cout<<task_id<<"\t"<<task_gain<<"\t"<<task_gain_->at(nid).at(task_id)<<"\t"<<( node_task_inst_num_right_.at(nid).at(task_id) + node_task_inst_num_left_.at(nid).at(task_id) )<<"\n";
          }
        }
      }

      if (param_.xgboost_task_gain_output_flag==1 && gain_all_at_node>0){
          std::ofstream out(param_.output_file,std::ios::app);
          if (out.is_open()){

            out<<(neg_task_gain_sample_num/sample_num)<<'-'<<sample_num;

          }
          out<<",";
          out.close();
        }

      if (param_.xgboost_task_gain_output_flag==2 && gain_all_at_node>0){
          std::ofstream out(param_.output_file,std::ios::app);
          if (out.is_open()){
            if ( (neg_task_gain_sample_num/sample_num) <= threshold_ratio_R){
              out<<(neg_task_gain_sample_num/sample_num);
             }
             else{
               out<<0;
             }


          }
          out<<",";
          out.close();
        }

      if ( (neg_task_gain_sample_num/sample_num) > threshold_ratio_R){
        return true;
      }
      else{
        return false;
      }


    }

    inline bool WhenTaskSplitMeanRatio(std::vector<std::vector<float> > * task_gain_,int nid, float threshold_ratio_R){

      float mean_task_gain = 0;

      
      float neg_task_gain_sample_num=0;
      float sample_num=( node_inst_num_right_.at(nid) + node_inst_num_left_.at(nid) );
      float task_gain =0;
      

      // calculate the mean
      for (int task_id : tasks_list_){

          task_gain = task_gain_->at(nid).at(task_id);
          int task_sample_num = node_task_inst_num_right_.at(nid).at(task_id) + node_task_inst_num_left_.at(nid).at(task_id);
          mean_task_gain += (task_sample_num*task_gain);
      }
      mean_task_gain/=sample_num;

      for (int task_id : tasks_list_){

          task_gain = task_gain_->at(nid).at(task_id);
          if (task_gain < (param_.mean_less_ratio*mean_task_gain)){
            int task_sample_num = node_task_inst_num_right_.at(nid).at(task_id) + node_task_inst_num_left_.at(nid).at(task_id);
            neg_task_gain_sample_num+=task_sample_num;
          }
      }
      if ( (neg_task_gain_sample_num/sample_num) > threshold_ratio_R){
        return true;
      }
      else{
        return false;
      }
    }

    //when the gain ratio of the task with negative task gain (task_gain_all only) is higher that some fixed ratio, we will set this node as a task split node
    inline bool WhenTaskSplitNegativeGainRatio(std::vector<std::vector<float> > * task_gain_,int nid, float threshold_ratio_R){
      
      float neg_task_gain=0;
      float gain_all = 0;

      
      for (int task_id : tasks_list_){
        float task_gain = task_gain_->at(nid).at(task_id);
        gain_all +=std::fabs(task_gain);

        if (task_gain<0){
          neg_task_gain += std::fabs(task_gain);
        }

        // if (param_.debug==12){
        //     std::cout<<task_id<<"\t"<<task_gain<<"\t"<<task_gain_->at(nid).at(task_id)<<"\t"<<( node_task_inst_num_right_.at(nid).at(task_id) + node_task_inst_num_left_.at(nid).at(task_id) )<<"\n";
        //   }
      }
      if (param_.debug==12){
            std::cout<<"\n========================================\n"<<neg_task_gain<<"   "<<gain_all<<"   "<<neg_task_gain/gain_all<<"\n";
          }

      // note that gain_all may be zero, because the tree may automatically split the tasks, and task gain for each task is 0
      // and we do not want a task split any more if the tree has already done it for us.
      if (gain_all<1e-6){
        return false;
      }

      if ( (neg_task_gain/gain_all) > threshold_ratio_R){
        return true;
      }
      else{
        return false;
      }
    }

    inline void HowTaskSplitHardMargin(std::vector<std::vector<float> > * task_gain_,const std::vector<int> &qexpand){
      for (int nid : qexpand){
        if (is_task_node_.at(nid)){
          for (int task_id : tasks_list_){
            if ((task_gain_->at(nid).at(task_id)-0)<param_.task_gain_margin) {
              task_node_left_tasks_.at(nid).push_back(task_id);
              }
            else{
              task_node_right_tasks_.at(nid).push_back(task_id);
            } 
          }
        }
      }
    }

    inline void CalPosRatio(const std::vector<int> &qexpand, 
                                            //const 
                                            DMatrix& fmat,
                                            std::vector<bst_uint> feat_set){
      // 1. calculate the pos ratio for each task on each node and also the node
      std::vector<bst_float> inst_label = fmat.Info().labels_;
      dmlc::DataIter<ColBatch> *iter_task = fmat.ColIterator(feat_set);
      
      // sum the pos num
      while (iter_task->Next()) {  
        const ColBatch &batch = iter_task->Value();
          ColBatch::Inst col = batch[0];
          const auto ndata = static_cast<bst_omp_uint>(col.length);
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = DecodePosition(ridx);  // HINT: remember this mistake, don't trust the name of data
            const int task_id = inst_task_id_.at(ridx);
            if (inst_label.at(ridx)==1){
              node_task_pos_ratio_.at(nid).at(task_id)+=1;
              node_pos_ratio_.at(nid)+=1;
            }
          }
      }

      // calculate the pos ratio
      for (int nid: qexpand){
        int node_sum = node_inst_num_left_.at(nid)+node_inst_num_right_.at(nid);
        if (node_sum>0){
          node_pos_ratio_.at(nid)/=node_sum;
        }
        else{
          node_pos_ratio_.at(nid)=0;
        }
        for (int task_id : tasks_list_){
          int sum = node_task_inst_num_left_.at(nid).at(task_id)+node_task_inst_num_right_.at(nid).at(task_id);
          float tmp_posnum=node_task_pos_ratio_.at(nid).at(task_id);
          node_task_pos_ratio_.at(nid).at(task_id) = ( tmp_posnum + param_.baseline_alpha * node_pos_ratio_.at(nid) )/( sum + param_.baseline_alpha );
        }
      }
    }


    inline void FindTaskSplitNode(const std::vector<int> &qexpand,RegTree *tree, const int depth){
      std::vector<std::vector<float> > * task_gain_;
      // first we need to claim which task gain we are using, task_gain_self_ or task_gain_all_ 
      if (param_.use_task_gain_self==1){
        task_gain_ = &task_gain_self_;
      }
      else{
        task_gain_ = &task_gain_all_;
      }

      // fir case 3,  we will only use task_gain_all 
      if (param_.when_task_split == 3){
        task_gain_ = &task_gain_all_;
      }

      if (depth < (param_.max_depth - param_.conduct_task_split_N_layer_before)){

        for (int nid : qexpand){
          // we should set a param here to indicate which function we are using to make a task split decision

          // 1. when there are some negative task split 
          switch (param_.when_task_split) {
            // TODO:
            case 0: is_task_node_.at(nid) = WhenTaskSplitHardMargin(task_gain_,nid,param_.min_task_gain); break;
            case 1: is_task_node_.at(nid) = WhenTaskSplitNegativeSampleRatio(task_gain_,nid,param_.threshold_ratio_R); break;
            case 2: is_task_node_.at(nid) = WhenTaskSplitRandom(); break;
            case 3: is_task_node_.at(nid) = WhenTaskSplitNegativeGainRatio(task_gain_,nid,param_.threshold_ratio_R); break;
            case 4: is_task_node_.at(nid) = WhenTaskSplitMeanRatio(task_gain_,nid,param_.threshold_ratio_R); break;
            
            
            case 9: break;  // when and how , pos ratio 
            case 10: is_task_node_.at(nid) = false; WhenTaskSplitNegativeSampleRatio(task_gain_,nid,param_.threshold_ratio_R); break;
            // case 1: CLIDumpModel(param); break;`
            // case 2: CLIPredict(param); break;`
          }
        }
      }
    }

    inline void ConductTaskSplit(const std::vector<int> &qexpand,
                                  RegTree *tree, 
                                  DMatrix *p_fmat,
                                  std::vector<bst_uint> feat_set,
                                  const std::vector<GradientPair> &gpair){
                                    


      std::vector<std::vector<float> > * task_gain_;
      // first we need to claim which task gain we are using, task_gain_self_ or task_gain_all_ 
      if (param_.use_task_gain_self==1){
        task_gain_ = &task_gain_self_;
      }
      else{
        task_gain_ = &task_gain_all_;
      }
      // TODO:

      switch (param_.how_task_split) {
        case 0: HowTaskSplitHardMargin(task_gain_,qexpand); break;
        case 1: HowTaskSplitOneLevelForward(task_gain_,qexpand,*p_fmat,feat_set,gpair); break;
        case 8: WhenAndHowTaskSplitRandom(task_gain_,qexpand,*p_fmat,feat_set,gpair); break;
        case 9: WhenAndHowTaskSplitPosRatio(task_gain_,qexpand,*p_fmat,feat_set,gpair); break;
        }
    }



    inline void WhenAndHowTaskSplitRandom(std::vector<std::vector<float> > * task_gain_,
                                            const std::vector<int> &qexpand_p, 
                                            //const 
                                            DMatrix& fmat,
                                            std::vector<bst_uint> feat_set,
                                            const std::vector<GradientPair>& gpair){

                                                
      

      std::vector<int> qexpand;
      qexpand.clear();
      for (int nid : qexpand_p){
        float ran_float = rand()*1.0/(RAND_MAX*1.0);
        if (param_.debug==10){
          std::cout<<nid<<"("<<ran_float<<","<<param_.baseline_lambda<<")"<<"\t";
        }
        if (ran_float < param_.baseline_lambda){
          qexpand.push_back(nid);
        }
      }
      if (qexpand.size() == 0) return;


      // 2. compare and find the highest task split gain 
      auto biggest = std::max_element(std::begin(qexpand), std::end(qexpand));  // FIXME: may have problem here.
      int num_node = (*biggest+1);

      
      std::vector<std::vector<std::pair<int, float> > > node_task_value_map;
      node_task_value_map.clear();
      node_task_value_map.resize(num_node);
      for (int nid : qexpand){
        for (int task_id : tasks_list_){
          node_task_value_map.at(nid).push_back(std::make_pair( task_id, rand()*1.0/(RAND_MAX*1.0) ) );
        }
      // 2.1. sort the tasks in each node
        std::sort(node_task_value_map.at(nid).begin(), node_task_value_map.at(nid).end() ,cmp);
      }

      //3 init some vals

      // 3.1 best_task_gain
      std::vector<float> best_task_gain;
      const float negative_infinity = -std::numeric_limits<float>::infinity();
      best_task_gain.resize(num_node,negative_infinity);

      //3.2 best left_tasks & right_tasks for each task node
      std::vector<std::vector<int> > task_node_left_tasks_best;
      std::vector<std::vector<int> > task_node_right_tasks_best;
      task_node_left_tasks_best.resize(num_node);
      task_node_right_tasks_best.resize(num_node);

      // 3.2 the sum G and H of left and right tasks
      std::vector<float> G_node_left_;
      std::vector<float> G_node_right_;
      std::vector<float> H_node_left_;
      std::vector<float> H_node_right_;

      G_node_left_.resize(num_node,0);
      G_node_right_.resize(num_node,0);
      H_node_left_.resize(num_node,0);
      H_node_right_.resize(num_node,0);

      for (int nid : qexpand){
        G_node_right_.at(nid) = G_node_.at(nid);
        H_node_right_.at(nid) = H_node_.at(nid);
      }



      for (int split_n =0;split_n <task_num_for_OLF-1;split_n++){ 
        for (int nid : qexpand){
          // remove one task from the right child to left.
          int task_id = node_task_value_map.at(nid).at(split_n).first;

          G_node_left_.at(nid) += (G_task_lnode_.at(nid).at(task_id)+G_task_rnode_.at(nid).at(task_id));
          G_node_right_.at(nid) -= (G_task_lnode_.at(nid).at(task_id)+G_task_rnode_.at(nid).at(task_id));
          H_node_left_.at(nid) += (H_task_lnode_.at(nid).at(task_id)+H_task_rnode_.at(nid).at(task_id));
          H_node_right_.at(nid) -= (H_task_lnode_.at(nid).at(task_id)+H_task_rnode_.at(nid).at(task_id));

          // if the value is the same, we will omit this split point. but we should make sure the sum G and H are correctlly calculated
          if (split_n>0 && node_task_value_map.at(nid).at(split_n).second == node_task_value_map.at(nid).at(split_n-1).second){
            continue;
          }

          float gain_left = Square(G_node_left_.at(nid)) / (H_node_left_.at(nid)+param_.reg_lambda);
          float gain_right = Square(G_node_right_.at(nid)) / (H_node_right_.at(nid)+param_.reg_lambda);
          float gain = gain_left+gain_right;


          // update the best task split gain and task
          if ( gain > best_task_gain.at(nid)  ){
            best_task_gain.at(nid) = gain;
            
            task_node_left_tasks_best.at(nid).clear();
            task_node_right_tasks_best.at(nid).clear();

            for ( int i =0 ; i < task_num_for_OLF ; i++){
              int task_id = node_task_value_map.at(nid).at(i).first;
              if (i <= split_n){
                task_node_left_tasks_best.at(nid).push_back( task_id );
              }
              else{
                task_node_right_tasks_best.at(nid).push_back( task_id );
              }
            }
          }
        }
      }

      // 4. calculate the feature split gain
      for (int nid : qexpand){
      
        float G_left=0;
        float G_right=0;
        float H_left=0;
        float H_right=0;
        for (int task_id: tasks_list_){
          G_left+= G_task_lnode_.at(nid).at(task_id);
          G_right+= G_task_rnode_.at(nid).at(task_id);
          H_left+= H_task_lnode_.at(nid).at(task_id);
          H_right+= H_task_rnode_.at(nid).at(task_id);
        }

        float gain_left = Square( G_left ) / ( H_left + param_.reg_lambda );
        float gain_right = Square( G_right ) / ( H_right + param_.reg_lambda );
        float feature_gain = gain_left + gain_right;
        // 5. make the task split node decision 


        if (feature_gain<best_task_gain.at(nid)){
          is_task_node_.at(nid)=true;
          task_node_left_tasks_.at(nid).clear();
          task_node_right_tasks_.at(nid).clear();
          for (int task_id : task_node_left_tasks_best.at(nid)){
            task_node_left_tasks_.at(nid).push_back(task_id);
          }
          for (int task_id : task_node_right_tasks_best.at(nid)){
            task_node_right_tasks_.at(nid).push_back(task_id);
          }
        }
      }
    }




    inline void WhenAndHowTaskSplitPosRatio(std::vector<std::vector<float> > * task_gain_,
                                            const std::vector<int> &qexpand_p, 
                                            //const 
                                            DMatrix& fmat,
                                            std::vector<bst_uint> feat_set,
                                            const std::vector<GradientPair>& gpair){

                                                
      

      std::vector<int> qexpand;
      qexpand.clear();
      for (int nid : qexpand_p){
        float ran_float = rand()*1.0/(RAND_MAX*1.0);
        if (param_.debug==10){
          std::cout<<nid<<"("<<ran_float<<","<<param_.baseline_lambda<<")"<<"\t";
        }
        if (ran_float < param_.baseline_lambda){
          qexpand.push_back(nid);
        }
      }
      if (qexpand.size() == 0) return;

      CalPosRatio(qexpand,fmat,feat_set);

      // 2. compare and find the highest task split gain 
      auto biggest = std::max_element(std::begin(qexpand), std::end(qexpand));  // FIXME: may have problem here.
      int num_node = (*biggest+1);

      
      std::vector<std::vector<std::pair<int, float> > > node_task_value_map;
      node_task_value_map.clear();
      node_task_value_map.resize(num_node);
      for (int nid : qexpand){
        for (int task_id : tasks_list_){
          node_task_value_map.at(nid).push_back(std::make_pair( task_id, node_task_pos_ratio_.at(nid).at(task_id) ) );
        }
      // 2.1. sort the tasks in each node
        std::sort(node_task_value_map.at(nid).begin(), node_task_value_map.at(nid).end() ,cmp);
      }

      //3 init some vals

      // 3.1 best_task_gain
      std::vector<float> best_task_gain;
      const float negative_infinity = -std::numeric_limits<float>::infinity();
      best_task_gain.resize(num_node,negative_infinity);

      //3.2 best left_tasks & right_tasks for each task node
      std::vector<std::vector<int> > task_node_left_tasks_best;
      std::vector<std::vector<int> > task_node_right_tasks_best;
      task_node_left_tasks_best.resize(num_node);
      task_node_right_tasks_best.resize(num_node);

      // 3.2 the sum G and H of left and right tasks
      std::vector<float> G_node_left_;
      std::vector<float> G_node_right_;
      std::vector<float> H_node_left_;
      std::vector<float> H_node_right_;

      G_node_left_.resize(num_node,0);
      G_node_right_.resize(num_node,0);
      H_node_left_.resize(num_node,0);
      H_node_right_.resize(num_node,0);

      for (int nid : qexpand){
        G_node_right_.at(nid) = G_node_.at(nid);
        H_node_right_.at(nid) = H_node_.at(nid);
      }



      for (int split_n =0;split_n <task_num_for_OLF-1;split_n++){ 
        for (int nid : qexpand){
          // remove one task from the right child to left.
          int task_id = node_task_value_map.at(nid).at(split_n).first;

          G_node_left_.at(nid) += (G_task_lnode_.at(nid).at(task_id)+G_task_rnode_.at(nid).at(task_id));
          G_node_right_.at(nid) -= (G_task_lnode_.at(nid).at(task_id)+G_task_rnode_.at(nid).at(task_id));
          H_node_left_.at(nid) += (H_task_lnode_.at(nid).at(task_id)+H_task_rnode_.at(nid).at(task_id));
          H_node_right_.at(nid) -= (H_task_lnode_.at(nid).at(task_id)+H_task_rnode_.at(nid).at(task_id));

          // if the value is the same, we will omit this split point. but we should make sure the sum G and H are correctlly calculated
          if (split_n>0 && node_task_value_map.at(nid).at(split_n).second == node_task_value_map.at(nid).at(split_n-1).second){
            continue;
          }

          float gain_left = Square(G_node_left_.at(nid)) / (H_node_left_.at(nid)+param_.reg_lambda);
          float gain_right = Square(G_node_right_.at(nid)) / (H_node_right_.at(nid)+param_.reg_lambda);
          float gain = gain_left+gain_right;


          // update the best task split gain and task
          if ( gain > best_task_gain.at(nid)  ){
            best_task_gain.at(nid) = gain;
            
            task_node_left_tasks_best.at(nid).clear();
            task_node_right_tasks_best.at(nid).clear();

            for ( int i =0 ; i < task_num_for_OLF ; i++){
              int task_id = node_task_value_map.at(nid).at(i).first;
              if (i <= split_n){
                task_node_left_tasks_best.at(nid).push_back( task_id );
              }
              else{
                task_node_right_tasks_best.at(nid).push_back( task_id );
              }
            }
          }
        }
      }

      // 4. calculate the feature split gain
      for (int nid : qexpand){
      
        float G_left=0;
        float G_right=0;
        float H_left=0;
        float H_right=0;
        for (int task_id: tasks_list_){
          G_left+= G_task_lnode_.at(nid).at(task_id);
          G_right+= G_task_rnode_.at(nid).at(task_id);
          H_left+= H_task_lnode_.at(nid).at(task_id);
          H_right+= H_task_rnode_.at(nid).at(task_id);
        }

        float gain_left = Square( G_left ) / ( H_left + param_.reg_lambda );
        float gain_right = Square( G_right ) / ( H_right + param_.reg_lambda );
        float feature_gain = gain_left + gain_right;
        // 5. make the task split node decision 


        if (feature_gain<best_task_gain.at(nid)){
          is_task_node_.at(nid)=true;
          task_node_left_tasks_.at(nid).clear();
          task_node_right_tasks_.at(nid).clear();
          for (int task_id : task_node_left_tasks_best.at(nid)){
            task_node_left_tasks_.at(nid).push_back(task_id);
          }
          for (int task_id : task_node_right_tasks_best.at(nid)){
            task_node_right_tasks_.at(nid).push_back(task_id);
          }
        }
      }
    }

    inline void SetRandTaskSplit(const int num_node,const std::vector<int> &qexpand){
        node_task_random_.clear();
        node_task_random_.resize(num_node);
        for (int nid : qexpand){
          node_task_random_.at(nid).resize(task_num_for_init_vec);
          for (int task_id : tasks_list_){
            node_task_random_.at(nid).at(task_id)=rand()*1.0/(RAND_MAX*1.0);
          }
        }
      }

    inline void HowTaskSplitOneLevelForward(std::vector<std::vector<float> > * task_gain_,
                                            const std::vector<int> &qexpand, 
                                            //const 
                                            DMatrix& fmat,
                                            std::vector<bst_uint> feat_set,
                                            const std::vector<GradientPair>& gpair){

      
      

                                               
      auto num_row=fmat.Info().num_row_;
      
      // stores the expand nid for task node and feature node
      std::vector<int> task_expand;
      // std::vector<int> feature_expand;
      task_expand.clear();
      
      
      for (int nid : qexpand){
        if (is_task_node_.at(nid)){
          task_expand.push_back(nid);
        }
      }
      if (param_.debug==11){
        std::cout<<"1! here !\t";
      }
  

      auto biggest = std::max_element(std::begin(qexpand), std::end(qexpand));  // FIXME: may have problem here.
      int num_node = (*biggest+1);

      num_node_for_task_ = num_node;
      // std::cout<<num_node_for_task_<<"\t";

      // we shoud add a bool vector to indicate whether a nid is in task_expand or not 
      std::vector<bool> is_task_expand_nid;
      is_task_expand_nid.resize(num_node,false);
      for (int nid : task_expand){
        is_task_expand_nid.at(nid)=true;
      }
      
      

      // 1. construc the specific value to make such task split. // should be the value passed to this function
      std::vector<std::vector<float> > * node_task_value;
      switch(param_.which_task_value){
        case 0: node_task_value = & task_w_; break; 
        case 1: node_task_value = & task_gain_self_; break;
        case 2: node_task_value = & task_gain_all_; break;
        case 3: CalPosRatio(task_expand,fmat,feat_set); node_task_value = & node_task_pos_ratio_; break;
        case 9: SetRandTaskSplit(num_node,task_expand); node_task_value = & node_task_random_; break;

      }

      // the <int, float> pair is sorted by the second value. 
      std::vector<std::vector<std::pair<int, float> > > node_task_value_map;
      node_task_value_map.clear();
      node_task_value_map.resize(num_node);
      for (int nid : task_expand){
        for (int task_id : tasks_list_){
          node_task_value_map.at(nid).push_back(std::make_pair( task_id, (*node_task_value).at(nid).at(task_id) ) );
        }
      // 2. sort the tasks in each node
        std::sort(node_task_value_map.at(nid).begin(), node_task_value_map.at(nid).end() ,cmp);
      } 
      

      // 3. init several vals used here
          //3.1 reserve space for the std::vector< std::vector<ThreadEntry> > task_stemp_; for we will node to
          // std::vector< std::vector<ThreadEntry> > task_stemp_;
          /////////////////////////////////////
          // task_stemp_.clear();
          // task_stemp_.resize(nthread_, std::vector<ThreadEntry>());
          // for (size_t i = 0; i < task_stemp_.size(); ++i) {
          //   task_stemp_[i].clear(); task_stemp_[i].resize(num_node*2+2,ThreadEntry(param_));
          // }//FIXME: not sure if the codes here work or not.
          ///////////////////////////////////////

          //3.25 the task split gain, we need to use it 

          //3.2 best one_level_forward (OLF) gain for each task node
          std::vector<float> best_OLF_gain;
          const float negative_infinity = -std::numeric_limits<float>::infinity();
          best_OLF_gain.resize(num_node,negative_infinity);

          //3.3 best left_tasks & right_tasks for each task node
          std::vector<std::vector<int> > task_node_left_tasks_best;
          std::vector<std::vector<int> > task_node_right_tasks_best;
          task_node_left_tasks_best.resize(num_node);
          task_node_right_tasks_best.resize(num_node);
          // for (int nid : task_expand){
          //     // TODO: should we resize and make the new vec for each node at each point?
          // }

          //3.4 a nid vector for each inst, this should also be updated after each round
          // for the task node nid x, we set the tmp left child nid as 2x-1, and the tmp right nid as 2x
          std::vector<int> position_task;
          position_task.resize(num_row,-1);
          

          // // the nid of each inst
          // // data_in
          // inst_nid_.resize(num_row_,-1);
          // // the task_id of each isnt
          // inst_task_id_.resize(num_row_,-1);
          // // whether the inst goes left (true) or right (false)
          // inst_go_left_.resize(num_row_,true);


          //3.5 a nid indexed dict used to indicate whether this node will be considered into the best split searching or not. 
          //    note that there should be two kinds of nid should be set apart. the first one is those feature split node, and 
          //    the second one is such node that in this round, there is no more task can be used to move from right to left
          //    ( or we can implement it as if the moved task is actually empty, which could be obtained by sum node_task_inst_num_left_ 
          //    and node_task_inst_num_right_ together )
          std::vector<bool> node_works;
          node_works.resize(num_node,false);
          

          //3.6 a vector that stores the temp nid of each task at each nid
          std::vector<std::vector<int> > sub_nid_of_task;
          sub_nid_of_task.resize(num_node);

          //3.7 the best node entry
          // std::vector<NodeEntry> snode_task_;
          // std::vector<TConstraint> constraints_task_
          // /////////////////////////////////////////////////////////////////
          // task_stemp_.clear();
          // task_stemp_.resize(nthread_, std::vector<ThreadEntry>());
          // for (size_t i = 0; i < task_stemp_.size(); ++i) {
          //   task_stemp_[i].clear(); task_stemp_[i].reserve(num_node*2+2,ThreadEntry(param_));
          // }//FIXME: not sure if the codes here work or not.


          // snode_task.resize(num_node*2+2,NodeEntry(param_));
          // constraints_task_.resize(num_node*2+2);
          // const RowSet &rowset = fmat.BufferedRowset();
          // const MetaInfo& info = fmat.Info();
          // // setup position
          // const auto ndata = static_cast<bst_omp_uint>(rowset.Size());
          // #pragma omp parallel for schedule(static)
          // for (bst_omp_uint i = 0; i < ndata; ++i) {
          //   const bst_uint ridx = rowset[i];
          //   const int tid = omp_get_thread_num();
          //   if (position_[ridx] < 0) continue;
          //   stemp_[tid][position_[ridx]].stats.Add(gpair, info, ridx);
          // }


          // ////////////////////////////////////////////////////////////////
          for (int nid : task_expand){
            node_works.at(nid)=true;

            sub_nid_of_task.at(nid).resize(task_num_for_init_vec,-1);
            for (int task_id : tasks_list_){
              sub_nid_of_task.at(nid).at(task_id) = GetRightChildNid(nid);
            }

          }

          //3.8 a sub nid indexed dict to indicate whether this sub nid is working
          std::vector<bool> sub_node_works;
          sub_node_works.resize(num_node*2+2,false);

          // 3.9 stores the gain of each sub_nid
          std::vector<float> sub_node_gain;
          sub_node_gain.resize(num_node*2+2,0);


          // 3.10 stores the G and H of each sub nid
          std::vector<float> sub_node_G;
          sub_node_G.resize(num_node*2+2,0);
          std::vector<float> sub_node_H;
          sub_node_H.resize(num_node*2+2,0);
          for ( int nid : task_expand){
            sub_node_G.at(GetRightChildNid(nid))=G_node_.at(nid);
            sub_node_H.at(GetRightChildNid(nid))=H_node_.at(nid);
          }

          //3.11 stores the pre gain of each nid
          std::vector<float> gain_nid;
          gain_nid.resize(num_node,0);



          

      dmlc::DataIter<ColBatch> *iter_task = fmat.ColIterator(feat_set);
      const ColBatch &batch = iter_task->Value();


      if (param_.debug==11){
        std::cout<<"2! here !\t";
      }


      // 4. find the best split for the child nodes, one task move at each time
      for (int split_n =0;split_n <task_num_for_OLF-1;split_n++){ // while there is remaining tasks to be moved from right to left (loop) 
      // for (int split_n =0;split_n <task_num_for_OLF;split_n++){ // while there is remaining tasks to be moved from right to left (loop) 

            if (param_.debug==11){
        std::cout<<"3! here !\t";
      }
      
          //3.9 a sub nid set 
          std::vector<int> sub_qexpand;

          // 4.1 move one task from right to left, update the ind_ridx vector, if the task is actually empty, we nid to set a flag to that nid
          // update the sub_nid_of_task.
          sub_qexpand.clear();
          for (int nid : task_expand){
            // remove one task from the right child to left.
            int task_id = node_task_value_map.at(nid).at(split_n).first;
            sub_nid_of_task.at(nid).at(task_id)=GetLeftChildNid(nid);
            // if the moved task is empty at this node, then we will do nothing with this node.
            if ( (node_task_inst_num_left_.at(nid).at(task_id) + node_task_inst_num_right_.at(nid).at(task_id) )<1 ){
              node_works.at(nid)=false;
              sub_node_works.at(GetLeftChildNid(nid))=false;
              sub_node_works.at(GetRightChildNid(nid))=false;
            }
            else{
              node_works.at(nid)=true;
              sub_node_works.at(GetLeftChildNid(nid))=true;
              sub_node_works.at(GetRightChildNid(nid))=true;
              sub_qexpand.push_back(GetLeftChildNid(nid));
              sub_qexpand.push_back(GetRightChildNid(nid));

            }

            // std::cout<<node_task_value_map.at(nid).at(split_n).second<<" value "<<split_n<<" split_n "<<task_id<<" task_id "<<nid<<" nid "<<(node_task_inst_num_left_.at(nid).at(task_id) + node_task_inst_num_right_.at(nid).at(task_id))<<"===\n";
          }


                if (param_.debug==11){
        std::cout<<"4! here !\t";
      }
          // update the sub-nid for each inst
          for (bst_uint ridx=0;ridx<num_row;ridx++){
            int nid=inst_nid_.at(ridx);
            int task_id=inst_task_id_.at(ridx);
            if (is_task_node_.at(nid) && is_task_expand_nid.at(nid)){
              // position_task.at(ridx)=sub_nid_of_task.at(nid).at(task_id);  
              //FIXME: the bug is here. needs fix (20180508)
              // the resson is: a nid mgiht not be a task_qexpand now, but it is still recorded as a task node, so when it comes to that nid, the  sub_nid_of_task.at(nid) 
              // is not initialized. and that cause the bug. 
              int tmp=sub_nid_of_task.at(nid).at(task_id);
              position_task.at(ridx)= tmp;
            }
          }
          
            if (param_.debug==11){
                  std::cout<<"  5 ! here !\t";
                }


          // 4.2 init some vals
          /////////////////////////////////////////////////////////////////
          task_stemp_.clear();
          task_stemp_.resize(nthread_, std::vector<ThreadEntry>());
          for (size_t i = 0; i < task_stemp_.size(); ++i) {
            task_stemp_[i].clear(); task_stemp_[i].resize(num_node*2+2,ThreadEntry(param_,negative_infinity));
          }//FIXME: not sure if the codes here work or not.


          snode_task_.clear(); // important!
          snode_task_.resize(num_node*2+2,NodeEntry(param_,negative_infinity));
          constraints_task_.resize(num_node*2+2);
          const RowSet &rowset = fmat.BufferedRowset();
          const MetaInfo& info = fmat.Info();
          // setup position
          const auto ndata = static_cast<bst_omp_uint>(rowset.Size());
          #pragma omp parallel for schedule(static)     // FIXME: (OMP) 
          for (bst_omp_uint i = 0; i < ndata; ++i) {
            const bst_uint ridx = rowset[i];
            const int tid = omp_get_thread_num();
            const int nid = position_task[ridx];
            if (nid < 0) continue;
            if (!sub_node_works.at(nid)) continue;
            task_stemp_[tid][nid].stats.Add(gpair, info, ridx);
          }

                


          // sum the per thread statistics together
          for (int nid : sub_qexpand) {
            TStats stats(param_);
            for (size_t tid = 0; tid < stemp_.size(); ++tid) {
              stats.Add(task_stemp_[tid][nid].stats);
            }
            // update node statistics
            snode_task_[nid].stats = stats;
          }
          // // setup constraints before calculating the weight
          // for (int nid : sub_qexpand) {
          //   // if (tree[nid].IsRoot()) continue; //FIXME: todo
          //   if (nid%2==0)  const int pid = nid/2;
          //   else const int pid = (nid+1)/2;  //FIXME: what is constraints!?
          //   constraints_task_[pid].SetChild(param_, 0,
          //                             snode_task_[GetLeftChildNid(pid)].stats,
          //                             snode_task_[GetRightChildNid(pid)].stats,
          //                             &constraints_task_[GetLeftChildNid(pid)],
          //                             &constraints_task_[GetRightChildNid(pid)]);
          // }// FIXME: the node number should be changed. 

          // calculating the weights
          for (int nid : sub_qexpand) {
            snode_task_[nid].root_gain = static_cast<float>(
                constraints_task_[nid].CalcGain(param_, snode_task_[nid].stats));
            snode_task_[nid].weight = static_cast<float>(
                constraints_task_[nid].CalcWeight(param_, snode_task_[nid].stats));
          }



          ////////////////////////////////////////////////////////////////

          // 4.3 go through the whole dataframe, drop those whose nid flag is false, find the best split gain
          // start enumeration
          const auto nsize = static_cast<bst_omp_uint>(batch.size);   // nsize is the number of features, this means feature level parallel
          #if defined(_OPENMP)
          const int batch_size = std::max(static_cast<int>(nsize / this->nthread_ / 32), 1);
          #endif
          
          #pragma omp parallel for schedule(dynamic, batch_size)
          for (bst_omp_uint i = 0; i < nsize; ++i) {


            const bst_uint fid = batch.col_index[i];
            const int tid = omp_get_thread_num();
            const ColBatch::Inst c = batch[i];
            const bool ind = c.length != 0 && c.data[0].fvalue == c.data[c.length - 1].fvalue;
            if (param_.NeedForwardSearch(fmat.GetColDensity(fid), ind)) {
              this->EnumerateSplitTask(c.data, c.data + c.length, +1,
                                  fid, gpair, info, task_stemp_[tid],sub_qexpand,position_task,sub_node_works);
            }
            if (param_.NeedBackwardSearch(fmat.GetColDensity(fid), ind)) {
              this->EnumerateSplitTask(c.data + c.length - 1, c.data - 1, -1,
                                  fid, gpair, info, task_stemp_[tid],sub_qexpand,position_task,sub_node_works);
            }
          }
          

          this->SyncBestSolutionTask(sub_qexpand);
          // for (int nid : sub_qexpand){
          //   std::cout<<nid<<"\t"<<snode_task_[nid].best.loss_chg<<"\n";
          // }

          // 4.4 before we cal the whole OLF gain, we have to calcu the left child loss_chg and the right child loss_chg
          // also we can calcu the whole loss_chg of root pid, then we have the final OLF gain,
          // OLF gain = root loss_chg + // optional
          //            left tasks gain + right tasks gain  // this will be done in this 4.4 section
          //            left loss_chg + right loss // this is done in 4.3

          //FIXME: 
          // for (int nid : sub_qexpand){
          //   int pid=-1;
          //   if (nid%2==0)  pid = nid/2;
          //   else  pid = (nid+1)/2; 
          //   float G=0;
          //   float H=0;
          //   for (int task_id : tasks_list_){
          //     if (sub_nid_of_task.at(pid).at(task_id) == nid){
          //       G +=(G_task_lnode_.at(pid).at(task_id)+G_task_rnode_.at(pid).at(task_id));
          //       H +=(H_task_lnode_.at(pid).at(task_id)+H_task_rnode_.at(pid).at(task_id));
          //     }
          //   }
          //   std::cout<<G<<"\t"<<H<<"\n";

          //   sub_node_gain.at(nid) = (G*G)/(H+param_.reg_lambda);
          // }
          for (int nid : task_expand){
            int task_id = node_task_value_map.at(nid).at(split_n).first;

            int left_nid=GetLeftChildNid(nid);
            int right_nid=GetRightChildNid(nid);

            sub_node_G.at(left_nid)+=(G_task_lnode_.at(nid).at(task_id)+G_task_rnode_.at(nid).at(task_id));
            sub_node_H.at(left_nid)+=(H_task_lnode_.at(nid).at(task_id)+H_task_rnode_.at(nid).at(task_id));

            sub_node_G.at(right_nid)-=(G_task_lnode_.at(nid).at(task_id)+G_task_rnode_.at(nid).at(task_id));
            sub_node_H.at(right_nid)-=(H_task_lnode_.at(nid).at(task_id)+H_task_rnode_.at(nid).at(task_id));
            float left_gain = (sub_node_G.at(left_nid)*sub_node_G.at(left_nid))/(sub_node_H.at(left_nid)+param_.reg_lambda);
            float right_gain = (sub_node_G.at(right_nid)*sub_node_G.at(right_nid))/(sub_node_H.at(right_nid)+param_.reg_lambda);
            float gain = ( (sub_node_G.at(left_nid)+sub_node_G.at(right_nid)) * (sub_node_G.at(left_nid)+sub_node_G.at(right_nid)) 
                          / ( sub_node_H.at(left_nid) + sub_node_H.at(right_nid) + param_.reg_lambda  ) );
            gain_nid.at(nid)=left_gain+right_gain-gain;

            /////////////////////////////
            // int left_nid=GetLeftChildNid(nid);
            // int right_nid=GetRightChildNid(nid);
            // if (nid==8)
            // std::cout<<nid<<" node  "<<left_gain<<"|"<<snode_task_[left_nid].root_gain<<"  8888  "<<right_gain<<"|"<<snode_task_[right_nid].root_gain<<"\n";
            /////////////////////////////
          }


          // 4.5 find the OLF gain and update it
          for (int nid : task_expand){
            if (node_works.at(nid)){
              float cur_OLF_gain=0;
              int left_nid=GetLeftChildNid(nid);
              int right_nid=GetRightChildNid(nid);
              
              cur_OLF_gain+=gain_nid.at(nid);
              cur_OLF_gain+=snode_task_[left_nid].best.loss_chg;
              cur_OLF_gain+=snode_task_[right_nid].best.loss_chg;

              // if (nid==77){
              //   std::cout<<snode_task_[left_nid].best.loss_chg<<" "<<snode_task_[right_nid].best.loss_chg<<"  "<<cur_OLF_gain<<"\t"<<best_OLF_gain.at(nid)<<"\n";
              // }
              // pos_num_flag is used to make sure that both the left tasks and right tasks have at least one samples. otherwise the
              // tasks split might be split all the non-empty tasks to the left and the empty tasks to the right.
              bool left_pos_num_flag=false;
              bool right_pos_num_flag=false;
              


              if (cur_OLF_gain>best_OLF_gain.at(nid)){
                for (int task_id : tasks_list_){
                  if (sub_nid_of_task.at(nid).at(task_id) == left_nid){
                    if ( (node_task_inst_num_left_.at(nid).at(task_id)+node_task_inst_num_right_.at(nid).at(task_id)) >0 ){
                      left_pos_num_flag=true;
                    }
                  }
                  else{
                    if ( (node_task_inst_num_left_.at(nid).at(task_id)+node_task_inst_num_right_.at(nid).at(task_id)) >0 ){
                      right_pos_num_flag=true;
                    }
                  }
                }

                if (left_pos_num_flag && right_pos_num_flag){
                  best_OLF_gain.at(nid) = cur_OLF_gain;
                  task_node_left_tasks_best.at(nid).clear();
                  task_node_right_tasks_best.at(nid).clear();
                  for (int task_id : tasks_list_){
                    if (sub_nid_of_task.at(nid).at(task_id) == left_nid){
                      task_node_left_tasks_best.at(nid).push_back(task_id);
                    }
                    else{
                      task_node_right_tasks_best.at(nid).push_back(task_id);
                    }
                  }
                }
                // if (nid==77){
                //   std::cout<<cur_OLF_gain<<"\t"<<best_OLF_gain.at(nid)<<"\n";
                //   std::cout<<task_node_left_tasks_best.at(nid).size()<<" "<<task_node_right_tasks_best.at(nid).size()<<"\n";
                // }

              }
              // if (param_.debug == 1){

              //   if (nid==param_.nid_debug){
                  
              //     std::cout<<gain_nid.at(nid)<<"\t"<<snode_task_[left_nid].best.loss_chg<<"\t"<<snode_task_[right_nid].best.loss_chg<<"\n";
              //     // std::cout<<snode_task_[left_nid].best.loss_chg+snode_task_[right_nid].best.loss_chg<<"\n";
              //     std::cout<<cur_OLF_gain<<"\t"<<best_OLF_gain.at(nid)<<"\n";
              //   }
              // }

            }
          }


          // end loop
      }

      // 5. get the best OLF for each node, and find the best left tasks & right tasks.  
      for (int nid : task_expand){
        task_node_left_tasks_.at(nid).clear();
        task_node_right_tasks_.at(nid).clear();
        // if (nid==28){
        //           // std::cout<<cur_OLF_gain<<"\t"<<best_OLF_gain.at(nid)<<"\n";
        //           std::cout<<task_node_left_tasks_best.at(nid).size()<<" "<<task_node_right_tasks_best.at(nid).size()<<"\n";
        //         }
        
        for (int task_id : task_node_left_tasks_best.at(nid)){
          task_node_left_tasks_.at(nid).push_back(task_id);
        }
        for (int task_id : task_node_right_tasks_best.at(nid)){
          task_node_right_tasks_.at(nid).push_back(task_id);
        }

        // if (param_.debug == 1){
        //   if (nid==param_.nid_debug){
        //     for (int ii =0; ii<21; ++ii){
        //           int task_id = node_task_value_map.at(nid).at(ii).first;

        //     std::cout<<node_task_value_map.at(nid).at(ii).first<<"\t"<<
        //               node_task_value_map.at(nid).at(ii).second<<"\t"<<
        //               (node_task_inst_num_left_.at(nid).at(task_id) + node_task_inst_num_right_.at(nid).at(task_id) )<<"\t"<<
        //               G_task_rnode_.at(nid).at(task_id)+G_task_lnode_.at(nid).at(task_id)<<"\t"<<
        //               H_task_rnode_.at(nid).at(task_id)+H_task_lnode_.at(nid).at(task_id)<<"\n";
        //   // float task_w_.at(nid).at(task_id)= -G_task_rnode_.at(nid).at(task_id) / (H_task_rnode_.at(nid).at(task_id) +param_.reg_lambda)
        //     }
        //     std::cout<<nid<<"sdfg\n";
        //   std::cout<<task_node_left_tasks_.at(nid).size()<<" "<<task_node_right_tasks_.at(nid).size()<<"\n";
          
        //   }
        // }
        if (param_.debug == 1){
          if (nid==param_.nid_debug){
            for (int task_id : tasks_list_){


            std::cout<<task_id<<"\t"<<(node_task_inst_num_left_.at(nid).at(task_id) + node_task_inst_num_right_.at(nid).at(task_id) )<<"\t"<<
                      G_task_rnode_.at(nid).at(task_id)+G_task_lnode_.at(nid).at(task_id)<<"\t"<<
                      H_task_rnode_.at(nid).at(task_id)+H_task_lnode_.at(nid).at(task_id)<<"\n";
          // float task_w_.at(nid).at(task_id)= -G_task_rnode_.at(nid).at(task_id) / (H_task_rnode_.at(nid).at(task_id) +param_.reg_lambda)
            }
            std::cout<<nid<<"sdfg\n";
          std::cout<<task_node_left_tasks_.at(nid).size()<<" "<<task_node_right_tasks_.at(nid).size()<<"\n";
          
          }
        }
      }

    }


    static bool cmp(const std::pair<int, float>& x, const std::pair<int, float>& y)    
    {    
        return x.second < y.second;    
    }  

    inline int GetRightChildNid(const int nid){
      return nid;
    }
    inline int GetLeftChildNid(const int nid){
      return nid+num_node_for_task_;
    }

//TODO: this functions could be added later...
/*        // is the parent is already a task split node, then the node will not perform task split. 
        if (successive_task_split == 1){ 
          if (nid>0){ 
            int p_ind = (*tree)[nid].Parent(); 
              if (is_task_node_.at(p_ind)){ 
                all_positive_flag=true; 
              } 
          } 
        }
                if (successive_task_split==2){ 
          if (nid>0){ 
            int p_ind = (*tree)[nid].Parent(); 
            int p_rchild_ind = (*tree)[nid].LeftChild(); 
              if (is_task_node_.at(p_ind) && p_rchild_ind==nid){ 
                all_positive_flag=true; 
              } 
          } 
        } 
        
        
        
        */

    // find splits at current level, do split per level
    inline void FindSplit(int depth,
                          const std::vector<int> &qexpand,
                          const std::vector<GradientPair> &gpair,
                          DMatrix *p_fmat,
                          RegTree *p_tree) {
      std::vector<bst_uint> feat_set = feat_index_;
      if (param_.colsample_bylevel != 1.0f) {
        std::shuffle(feat_set.begin(), feat_set.end(), common::GlobalRandom());
        unsigned n = std::max(static_cast<unsigned>(1),
                              static_cast<unsigned>(param_.colsample_bylevel * feat_index_.size()));
        CHECK_GT(param_.colsample_bylevel, 0U)
            << "colsample_bylevel cannot be zero.";
        feat_set.resize(n);
      }

      // we should add the task feature if it is dropped by the colsample_bylevel
      std::vector<bst_uint>::iterator ret = std::find(feat_set.begin(), feat_set.end(), 0);
      if (ret==feat_set.end()){
        feat_set.push_back(0);
      }


      dmlc::DataIter<ColBatch>* iter = p_fmat->ColIterator(feat_set);
      while (iter->Next()) {
        this->UpdateSolution(iter->Value(), gpair, *p_fmat); //because it's level wise training
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      this->SyncBestSolution(qexpand);
      // get the best result, we can synchronize the solution



      //=====================  begin of task split =======================
 
      /* 1. calculate task_gain_all, task_gain_self, w*,
        * 2. partition the samples under some rule
        * 3. calculate the gain of left and right child if they conduct a normal feature split
        * 4. compare the 1-4 gain of all the task split and make a decision, do task or not.
        * 5. make a node, we need to modify the node data structure, add a flag `is_task_split_node`
        *    we also have to update several predicting functions that uses the tree structure.
        *
        * */

      /****************************** init auxiliary val ***********************************/
      auto num_row=p_fmat->Info().num_row_;
      auto biggest = std::max_element(std::begin(qexpand), std::end(qexpand));  // FIXME: may have problem here.
      int num_node = (*biggest+1);
      // if (param_.debug==1){
      //   for (int nid: qexpand){
      //     std::cout<<"~~~"<<nid<<"~~~\n";
      //   }
      //   std::cout<<"\n****"<<num_node<<"****\n";
      // }

      InitAuxiliary(num_row,num_node,qexpand);
      // std::cout<<"init aux"<<'\n';

      // /******************  assign nid,  task_id,  go_left  for each inst *******************/
      AssignTaskAndNid(p_tree,p_fmat,feat_set,qexpand,num_node);
      // for (int ind : inst_nid_){std::cout<<ind<<"\t";}
      // for (int ind : inst_task_id_){std::cout<<ind<<"\t";}
      // for (int ind : inst_go_left_){std::cout<<ind<<"\t";}

      /************************* calculate G, H through the whole data*************************/
      CalcGHInNode(num_row,gpair);

      /************* calculate task_gain_self and task_gain_self through the whole data********/
      CalcTaskWStar(qexpand);
      CalcTaskGainSelf(qexpand);
      // CalcTaskGainAll(qexpand);
      CalcTaskGainAllLambdaWeighted(qexpand);

      // /************* Calculate the task gain of each node ********/
      // AccumulateTaskGain(qexpand);


      /******************  make the decision whether make a task split or not   ****************/
      // the results are updated in the is_task_node_
      FindTaskSplitNode(qexpand,p_tree,depth);


      // before we conduct the task split things, we should make those node with only one task apart.
      for (int nid : qexpand){
        if (is_task_node_.at(nid)){
          int pos_num_task=0;
          for (int task_id : tasks_list_ ){
            if ( (node_task_inst_num_left_.at(nid).at(task_id) + node_task_inst_num_right_.at(nid).at(task_id) )>0){
              pos_num_task+=1;
            } 
          }
          if (pos_num_task<2){
            is_task_node_.at(nid)=false;
            task_node_pruned_num_++;


if (param_.debug==13){

            float neg_task_gain=0;
            float gain_all = 0;

      
      for (int task_id : tasks_list_){
        float task_gain = task_gain_all_.at(nid).at(task_id);
        gain_all +=std::fabs(task_gain);

        if (task_gain<0){
          neg_task_gain += std::fabs(task_gain);
        }

        std::cout<<task_id<<"\t"<<task_gain<<"\t"<<task_gain_all_.at(nid).at(task_id)<<"\t"<<( node_task_inst_num_right_.at(nid).at(task_id) + node_task_inst_num_left_.at(nid).at(task_id) )<<"\n";
      }
        std::cout<<"\n========================================\n"<<neg_task_gain<<"   "<<gain_all<<"   "<<neg_task_gain/gain_all<<"\n";
}
          }
        }
      }

      for (int nid : qexpand){
        if (is_task_node_.at(nid)){
          ConductTaskSplit(qexpand,p_tree,p_fmat,feat_set,gpair);
          break;
        }
      }


      if (param_.how_task_split==9 || param_.how_task_split==8 ){
          ConductTaskSplit(qexpand,p_tree,p_fmat,feat_set,gpair);        
      }

      if (param_.debug == 1){
        for (int nid : qexpand){

          if (nid==param_.nid_debug){
            for (int task_id : tasks_list_){


            std::cout<<task_id<<"\t"<<(node_task_inst_num_left_.at(nid).at(task_id) + node_task_inst_num_right_.at(nid).at(task_id) )<<"\t"<<
                      G_task_rnode_.at(nid).at(task_id)+G_task_lnode_.at(nid).at(task_id)<<"\t"<<
                      H_task_rnode_.at(nid).at(task_id)+H_task_lnode_.at(nid).at(task_id)<<"\n";
          // float task_w_.at(nid).at(task_id)= -G_task_rnode_.at(nid).at(task_id) / (H_task_rnode_.at(nid).at(task_id) +param_.reg_lambda)
            }
            std::cout<<nid<<"sdfg\n";
          std::cout<<task_node_left_tasks_.at(nid).size()<<" "<<task_node_right_tasks_.at(nid).size()<<"\n";
          
          }
        }
      }
      

      // if (param_.task_split_flag==1){
      //   ConductTaskSpl it(qexpand,p_tree);
      // }

      /******************  conduct task split based on the task_gain_all & sefl ****************/
      // bool task_split_flag=false;
      // bool task_split_flag=true;
      
      // // int successive_task_split=0;  // do nothing 
      // // int successive_task_split=1;  // the task split node's two child will not perform task split 
      // int successive_task_split=2;     // the task split node's right child will not perform task split  
      // int min_task_gain=param_.min_task_gain;
      // if (task_split_flag){
      //   // SimpleTaskGainAllSplit(qexpand,p_tree,successive_task_split,min_task_gain);
      //   SimpleTaskGainSelfSplit(qexpand,p_tree,successive_task_split,min_task_gain);
      // }

      // for (int nid : qexpand){
      //   for (int task_id : tasks_list_){
      //     // std::cout<<task_gain_self_.at(nid).at(task_id)<<"\t";
      //     std::cout<<task_gain_all_.at(nid).at(task_id)<<"\t";
      //     // std::cout<<task_w_.at(nid).at(task_id)<<"\t";
      //   }
      //   std::cout<<"\n";
      // }
      // std::cout<<"===========================================\n";
      
      //=====================  end   of task split =======================

        for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > kRtEps) {


          // we can only accumulate the task gain here, because these are those nodes have child nodes.
          if (!is_task_node_.at(nid)){
            for (int task_id : tasks_list_){
              accumulate_task_gain_.at(task_id)+=task_gain_all_.at(nid).at(task_id);
            }
          }
          
          p_tree->AddChilds(nid);
          // if (param_.task_split_flag==1){
          (*p_tree)[nid].SetSplitTask(e.best.SplitIndex(), 
                                    e.best.split_value,
                                    // left_tasks
                                    task_node_left_tasks_.at(nid),
                                    // right_tasks
                                    task_node_right_tasks_.at(nid),
                                    is_task_node_.at(nid),
                                    e.best.DefaultLeft()
                                    // is_task_split
                                    );
          
          if (!is_task_node_.at(nid)){
            // mark right child as 0, to indicate fresh leaf
            (*p_tree)[(*p_tree)[nid].LeftChild()].SetLeaf(0.0f, 0);
            (*p_tree)[(*p_tree)[nid].RightChild()].SetLeaf(0.0f, 0);
          }
          else{//FIXME: why did i set a else here? 
            (*p_tree)[(*p_tree)[nid].LeftChild()].SetLeaf(0.0f, 0);
            (*p_tree)[(*p_tree)[nid].RightChild()].SetLeaf(0.0f, 0);
          }
          
          if (param_.leaf_output_flag>0){
            // we will record the info of the leaf here
            // we first count the sample num of each task at this nid
            int all_num=0;
            std::ofstream out(param_.output_file,std::ios::app);
            auto num_row=p_fmat->Info().num_row_;
            std::vector<int> task_sample;
            task_sample.resize(task_num_for_init_vec,0);

            for (uint64_t ridx=0; ridx <num_row;ridx++){
              if (inst_nid_.at(ridx)==nid){
                int task_id = inst_task_id_.at(ridx);
                task_sample.at(task_id)+=1;
                all_num+=1;
              }
            }
            
            if (out.is_open()){
              if (!is_task_node_.at(nid)){
                out<<"f,";
              }
              else{
                out<<"t,";
              }
              out<<nid<<","<<snode_[nid].weight * param_.learning_rate<<","<<all_num<<",";

              if (param_.leaf_output_flag==2){
                for (int task_id :tasks_list_){
                  out<<task_id<<","<<task_sample.at(task_id)<<",";
                }
              }
            }
            out<<"tasks,";
            int sum_pos_nodes=0;
            for (int task_id:tasks_list_){
              if (task_sample.at(task_id)>0){
                out<<task_id<<",";
                sum_pos_nodes++;
              }
            }
            out<<"sum,"<<sum_pos_nodes<<",";
            out<<"\n";
            out.close();

          }
        } else {
          (*p_tree)[nid].SetLeaf(e.weight * param_.learning_rate);
          if (param_.leaf_output_flag>0){
            // we will record the info of the leaf here
            // we first count the sample num of each task at this nid
            int all_num=0;
            std::ofstream out(param_.output_file,std::ios::app);
            auto num_row=p_fmat->Info().num_row_;
            std::vector<int> task_sample;
            task_sample.resize(task_num_for_init_vec,0);

            for (uint64_t ridx=0; ridx <num_row;ridx++){
              if (inst_nid_.at(ridx)==nid){
                int task_id = inst_task_id_.at(ridx);
                task_sample.at(task_id)+=1;
                all_num+=1;
              }
            }
            
            if (out.is_open()){
              out<<"l,"<<nid<<","<<snode_[nid].weight * param_.learning_rate<<","<<all_num<<",";

              if (param_.leaf_output_flag==2){
                for (int task_id :tasks_list_){
                  out<<task_id<<","<<task_sample.at(task_id)<<",";
                }
              }
            }
            out<<"tasks,";
            int sum_pos_nodes=0;
            for (int task_id:tasks_list_){
              if (task_sample.at(task_id)>0){
                out<<task_id<<",";
                sum_pos_nodes++;
              }
            }
            out<<"sum,"<<sum_pos_nodes<<",";
            out<<"\n";
            out.close();

          }
        }
      }
    }
    // reset position of each data points after split is created in the tree
    inline void ResetPosition(const std::vector<int> &qexpand,
                              DMatrix* p_fmat,
                              const RegTree& tree) {
      // set the positions in the nondefault
      this->SetNonDefaultPosition(qexpand, p_fmat, tree);
      // set rest of instances to default position
      const RowSet &rowset = p_fmat->BufferedRowset();
      // set default direct nodes to default
      // for leaf nodes that are not fresh, mark then to ~nid,
      // so that they are ignored in future statistics collection
      const auto ndata = static_cast<bst_omp_uint>(rowset.Size());

      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        CHECK_LT(ridx, position_.size())
            << "ridx exceed bound " << "ridx="<<  ridx << " pos=" << position_.size();
        const int nid = this->DecodePosition(ridx);
        
        if (tree[nid].IsLeaf()) {
          // mark finish when it is not a fresh leaf
          if (tree[nid].RightChild() == -1) {
            position_[ridx] = ~nid;
          }
        } else {
          // push to default branch
          // if we can hold that all task feature will never be empty, we can leave the code here.
          if (tree[nid].DefaultLeft()) {
            this->SetEncodePosition(ridx, tree[nid].LeftChild());
          } else {
            this->SetEncodePosition(ridx, tree[nid].RightChild());
          }
        }
      }
    }
    // customization part
    // synchronize the best solution of each node
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        for (int tid = 0; tid < this->nthread_; ++tid) {
          e.best.Update(stemp_[tid][nid].best);
        }
      }
    }
     // synchronize the best solution of each node
    virtual void SyncBestSolutionTask(const std::vector<int> &qexpand) {
      for (int nid : qexpand) {
        NodeEntry &e = snode_task_[nid];
        for (int tid = 0; tid < this->nthread_; ++tid) {
          e.best.Update(task_stemp_[tid][nid].best);
        }
      }
    }
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       DMatrix *p_fmat,
                                       const RegTree &tree) { 
      // step 1, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        if (!tree[nid].IsLeaf()) {
          fsplits.push_back(tree[nid].SplitIndex());
        }
        // we need to add task feature here,
        if (is_task_node_.at(nid)){
          fsplits.push_back(0); // this may do
        }
      }

      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());

      dmlc::DataIter<ColBatch> *iter_temp = p_fmat->ColIterator(fsplits);
      const ColBatch &batch_temp = iter_temp->Value();
      ColBatch::Inst col = batch_temp[0];
      const auto ndata = static_cast<bst_omp_uint>(col.length);
      temp_position_.resize(ndata);
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = this->DecodePosition(ridx);
            temp_position_.at(ridx)=nid;
          }
      dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator(fsplits);
      while (iter->Next()) { 
        const ColBatch &batch = iter->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const auto ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)   
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = temp_position_.at(ridx);
            const bst_float fvalue = col[j].fvalue;
            /* HINT: the bug here is caused by the logic of getting nid of a ridx. after the firts round of position
            * assignment, some ridx has already change its nid in `position_`, then, when we vist the ridx for the second 
            * time, the nid might be one that has not resered space in position_
            * so, one solution is to remember the nid of each ridx before make the position assignment.
            * TODO: is there any other solution?
            * 
            */
            // go back to parent, correct those who are not default
            if (!(is_task_node_.at(nid))){ // for those who are feature split 
              if (!tree[nid].IsLeaf() && tree[nid].SplitIndex() == fid) {
                if (fvalue < tree[nid].SplitCond()) {
                  this->SetEncodePosition(ridx, tree[nid].LeftChild());
                } else {
                  this->SetEncodePosition(ridx, tree[nid].RightChild());
                }
              }
            }
            else{ // for those who are task split node  
              if (!tree[nid].IsLeaf() && fid==0){
                int task_value = int(fvalue);
                // search whether it is in the left tasks or not 
                std::vector<int>::iterator ret = std::find(task_node_left_tasks_.at(nid).begin(), task_node_left_tasks_.at(nid).end(), task_value);
                if (ret!=task_node_left_tasks_.at(nid).end()){
                  this->SetEncodePosition(ridx, tree[nid].LeftChild());
                }
                else {
                  // search whether it is in the right tasks or not 
                  std::vector<int>::iterator ret = std::find(task_node_right_tasks_.at(nid).begin(), task_node_right_tasks_.at(nid).end(), task_value);
                  if (ret!=task_node_right_tasks_.at(nid).end()){
                    this->SetEncodePosition(ridx, tree[nid].RightChild());
                  }
                  else{
                    for (int id :task_node_left_tasks_.at(nid)){
                      std::cout<<id<<"l\t";
                    }
                    for (int id :task_node_right_tasks_.at(nid)){
                      std::cout<<id<<"r\t";
                    }
                    exit(99);                    
                  }
                }
              }
            }
          }
        }
      }
    }
    // utils to get/set position, with encoded format
    // return decoded position
    inline int DecodePosition(bst_uint ridx) const {
      const int pid = position_[ridx];
      return pid < 0 ? ~pid : pid;
    }
    // encode the encoded position value for ridx
    inline void SetEncodePosition(bst_uint ridx, int nid) {
      if (position_[ridx] < 0) {
        position_[ridx] = ~nid;
      } else {
        position_[ridx] = nid;
      }
    }
    //  --data fields--
    const TrainParam& param_;
    // number of omp thread used during training
    const int nthread_;
    // Per feature: shuffle index of each feature index
    std::vector<bst_uint> feat_index_;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position_;
    // PerThread x PerTreeNode: statistics for per thread construction
    std::vector< std::vector<ThreadEntry> > stemp_;  //FIXME: this is important, we shoud check how it works. 
    // in this stemp_, each element is a vector for one thread. and this also means each thread is computing just one feature of the data.
    // then in the std::vector<ThreadEntry>, each Entry is just for one split node (expand_ node here)
    // the results of the best split gain and split condition is saved in this function. 



    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode_;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
    // constraint value
    std::vector<TConstraint> constraints_;


    /****************************** auxiliary val ***********************************/
    std::vector<int> exp_list_;
    // they can be init at the begining of each depth before the real task split

    // 
    std::vector<int> tasks_list_{1, 2, 4, 5, 6, 10, 11, 12, 13, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    int task_num_for_init_vec=30;    
    int task_num_for_OLF=21;


    // the nid of each inst
    std::vector<int> inst_nid_;

    // the task_id of each isnt
    std::vector<int> inst_task_id_;

    // whether the inst goes left (true) or right (false)
    std::vector<bool> inst_go_left_;



    // note that the index starts from 0
    // G and H in capital means the sum of the G and H over all the instances
    // store the G and H in the node, in order to calculate w* for the whole node 
    std::vector<float> G_node_;
    std::vector<float> H_node_;

    // store the G and H for each task in each node's left and right child, the 
    std::vector<std::vector<float> > G_task_lnode_;
    std::vector<std::vector<float> > G_task_rnode_;
    std::vector<std::vector<float> > H_task_lnode_;
    std::vector<std::vector<float> > H_task_rnode_; 

    // store the task gain in each nid for each task; 
    std::vector<std::vector<float> > task_gain_self_;
    std::vector<std::vector<float> > task_gain_all_;

    // store the task best w* in each node 
    std::vector<std::vector<float> > task_w_;

    // store the flag if the node nid is a task split node
    std::vector<bool> is_task_node_;

    // store the left and right task task_idx for each task split node 
    std::vector<std::vector<int> > task_node_left_tasks_;
    std::vector<std::vector<int> > task_node_right_tasks_;


    // store the temp positon
    std::vector<int> temp_position_;

    // store the inst num of each task at each node to calculate the weighted task_gain_all
    std::vector<std::vector<int> > node_task_inst_num_left_;
    std::vector<std::vector<int> > node_task_inst_num_right_;
    std::vector<int> node_inst_num_left_;
    std::vector<int> node_inst_num_right_;
    std::vector<int> node_inst_num_;

    // store the accumulate task gain of each task.
    std::vector<float> accumulate_task_gain_;




    

    /****************************** auxiliary val for 1-4 gain compute***********************************/
    // TODO:
    /* 
    * 1. note that we only conduct sucn 1-4 task split on the task split node already been determined to do task split.
    * 2. but we can also use this 1-4 gain to do the when to split task, todo.
    *  
    */
    // std::vector< std::vector<ThreadEntry> > stemp_ // FIXME:  find out when and how is stemp_ init. 
    //
    std::vector< std::vector<ThreadEntry> > task_stemp_;

    // t
    std::vector<NodeEntry> snode_task_;

    //
    std::vector<TConstraint> constraints_task_;

    //
    int num_node_for_task_;

    /****************************** auxiliary val for baseline***********************************/    
    // store the pos ratio of each task on each node
    std::vector<std::vector<float> > node_task_pos_ratio_;
    
    // store the pos ratio of each node
    std::vector<float> node_pos_ratio_;

    // // store the sum_G of left/right tasks 
    // std::vector<float> G_node_left_;
    // std::vector<float> G_node_right_;
    
    // // store the sum_H of left/right tasks
    // std::vector<float> H_node_left_;
    // std::vector<float> H_node_right_;
    //

    // to show that the task gain is usefull, we also sort the tasks randomly in OLF
    std::vector<std::vector<float> > node_task_random_; 

    int task_node_pruned_num_=0;
    
    

    
 

    
    
    
  };
};

// distributed column maker
template<typename TStats, typename TConstraint>
class DistColMaker : public ColMaker<TStats, TConstraint> {
 public:
  DistColMaker() : builder_(param_) {
    pruner_.reset(TreeUpdater::Create("prune"));
  }
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
    pruner_->Init(args);
  }
  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    TStats::CheckInfo(dmat->Info());
    CHECK_EQ(trees.size(), 1U) << "DistColMaker: only support one tree at a time";
    // build the tree
    builder_.Update(gpair->HostVector(), dmat, trees[0]);
    //// prune the tree, note that pruner will sync the tree
    pruner_->Update(gpair, dmat, trees);
    // update position after the tree is pruned
    builder_.UpdatePosition(dmat, *trees[0]);
  }

 private:
  class Builder : public ColMaker<TStats, TConstraint>::Builder {
   public:
    explicit Builder(const TrainParam &param)
        : ColMaker<TStats, TConstraint>::Builder(param) {}
    inline void UpdatePosition(DMatrix* p_fmat, const RegTree &tree) {
      const RowSet &rowset = p_fmat->BufferedRowset();
      const auto ndata = static_cast<bst_omp_uint>(rowset.Size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        int nid = this->DecodePosition(ridx);
        while (tree[nid].IsDeleted()) {
          nid = tree[nid].Parent();
          CHECK_GE(nid, 0);
        }
        this->position_[ridx] = nid;
      }
    }
    inline const int* GetLeafPosition() const {
      return dmlc::BeginPtr(this->position_);
    }

   protected:
    void SetNonDefaultPosition(const std::vector<int> &qexpand, DMatrix *p_fmat,
                               const RegTree &tree) override {
      // step 2, classify the non-default data into right places 
      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        if (!tree[nid].IsLeaf()) {
          fsplits.push_back(tree[nid].SplitIndex());
        }
      }
      // get the candidate split index
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      while (fsplits.size() != 0 && fsplits.back() >= p_fmat->Info().num_col_) {
        fsplits.pop_back();
      }
      // bitmap is only word concurrent, set to bool first
      {
        auto ndata = static_cast<bst_omp_uint>(this->position_.size());
        boolmap_.resize(ndata);
        #pragma omp parallel for schedule(static)
        for (bst_omp_uint j = 0; j < ndata; ++j) {
            boolmap_[j] = 0;
        }
      }
      dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator(fsplits);
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const auto ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const bst_float fvalue = col[j].fvalue;
            const int nid = this->DecodePosition(ridx);
            if (!tree[nid].IsLeaf() && tree[nid].SplitIndex() == fid) {
              if (fvalue < tree[nid].SplitCond()) {
                if (!tree[nid].DefaultLeft()) boolmap_[ridx] = 1;
              } else {
                if (tree[nid].DefaultLeft()) boolmap_[ridx] = 1;
              }
            }
          }
        }
      }

      bitmap_.InitFromBool(boolmap_);
      // communicate bitmap
      rabit::Allreduce<rabit::op::BitOR>(dmlc::BeginPtr(bitmap_.data), bitmap_.data.size());
      const RowSet &rowset = p_fmat->BufferedRowset();
      // get the new position
      const auto ndata = static_cast<bst_omp_uint>(rowset.Size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        const int nid = this->DecodePosition(ridx);
        if (bitmap_.Get(ridx)) {
          CHECK(!tree[nid].IsLeaf()) << "inconsistent reduce information";
          if (tree[nid].DefaultLeft()) {
            this->SetEncodePosition(ridx, tree[nid].RightChild());
          } else {
            this->SetEncodePosition(ridx, tree[nid].LeftChild());
          }
        }
      }
    }
    // synchronize the best solution of each node
    void SyncBestSolution(const std::vector<int> &qexpand) override {
      std::vector<SplitEntry> vec;
      for (int nid : qexpand) {
        for (int tid = 0; tid < this->nthread_; ++tid) {
          this->snode_[nid].best.Update(this->stemp_[tid][nid].best);
        }
        vec.push_back(this->snode_[nid].best);
      }
      // TODO(tqchen) lazy version
      // communicate best solution
      reducer_.Allreduce(dmlc::BeginPtr(vec), vec.size());
      // assign solution back
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        this->snode_[nid].best = vec[i];
      }
    }

   private:
    common::BitMap bitmap_;
    std::vector<int> boolmap_;
    rabit::Reducer<SplitEntry, SplitEntry::Reduce> reducer_;
  };
  // we directly introduce pruner here
  std::unique_ptr<TreeUpdater> pruner_;
  // training parameter
  TrainParam param_;
  // pointer to the builder
  Builder builder_;
};

// simple switch to defer implementation.
class TreeUpdaterSwitch : public TreeUpdater {
 public:
  TreeUpdaterSwitch()  = default;
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    for (auto &kv : args) {
      if (kv.first == "monotone_constraints" && kv.second.length() != 0) {
        monotone_ = true;
      }
    }
    if (inner_ == nullptr) {
      if (monotone_) {
        inner_.reset(new ColMaker<GradStats, ValueConstraint>());
      } else {
        inner_.reset(new ColMaker<GradStats, NoConstraint>());
      }
    }

    inner_->Init(args);
  }
  
  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* data,
              const std::vector<RegTree*>& trees) override {
    CHECK(inner_ != nullptr);
    inner_->Update(gpair, data, trees);
  }

 private:
  //  monotone constraints
  bool monotone_{false};
  // internal implementation
  std::unique_ptr<TreeUpdater> inner_;
};

XGBOOST_REGISTER_TREE_UPDATER(ColMaker, "grow_colmaker")
.describe("Grow tree with parallelization over columns.")
.set_body([]() {
    return new TreeUpdaterSwitch();
  });

XGBOOST_REGISTER_TREE_UPDATER(DistColMaker, "distcol")
.describe("Distributed column split version of tree maker.")
.set_body([]() {
    return new DistColMaker<GradStats, NoConstraint>();
  });
}  // namespace tree
}  // namespace xgboost
