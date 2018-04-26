/*!
 * Copyright 2014 by Contributors
 * \file updater_colmaker.cc
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */
#include <xgboost/tree_updater.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include "./param.h"
#include "../common/random.h"
#include "../common/bitmap.h"
#include "../common/sync.h"
#include <iostream>

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
  };
  // actual builder that runs the algorithm
  class Builder {
   public:
    // constructor
    explicit Builder(const TrainParam& param) : param_(param), nthread_(omp_get_max_threads()) {}
    // update one tree, growing
    virtual void Update(const std::vector<GradientPair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {
      this->InitData(gpair, *p_fmat, *p_tree);
      this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
      for (int depth = 0; depth < param_.max_depth; ++depth) {
        this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, &qexpand_);
        this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // set all the rest expanding nodes to leaf
      for (size_t i = 0; i < qexpand_.size(); ++i) {
        const int nid = qexpand_[i];
        (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
        p_tree->Stat(nid).base_weight = snode_[nid].weight;
        p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
        snode_[nid].stats.SetLeafVec(param_, p_tree->Leafvec(nid));
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
        snode_.resize(tree.param.num_nodes, NodeEntry(param_)); // FIXME: learn this, 
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
        temp[nid].stats.Clear();
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
                               // TODO:  what is ThreadEntry? this is a node?
      // use cacheline aware optimization
      if (TStats::kSimpleStats != 0 && param_.cache_opt != 0) {
        EnumerateSplitCacheOpt(begin, end, d_step, fid, gpair, temp);
        return;
      }
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats.Clear();
      }
      // left statistics
      TStats c(param_);
      for (const ColBatch::Entry *it = begin; it != end; it += d_step) {
        const bst_uint ridx = it->index;
        const int nid = position_[ridx];   // FIXME: this is the one that store which the data belongs to!
        if (nid < 0) continue;
        // start working
        const bst_float fvalue = it->fvalue;
        // get the statistics of nid
        ThreadEntry &e = temp[nid];    // TODO:  this one is important
        // test if first hit, this is fine, because we set 0 during init
        if (e.stats.Empty()) { 
          e.stats.Add(gpair, info, ridx);  // TODO: fixme:  important function   
          e.last_fvalue = fvalue;   // TODO: important function 
        } else {
          // try to find a split
          if (fvalue != e.last_fvalue &&
              e.stats.sum_hess >= param_.min_child_weight) {
            c.SetSubstract(snode_[nid].stats, e.stats); // TODO: important function
            if (c.sum_hess >= param_.min_child_weight) {
              bst_float loss_chg;
              if (d_step == -1) {
                loss_chg = static_cast<bst_float>(
                    constraints_[nid].CalcSplitGain( // TODO: important function
                        param_, param_.monotone_constraints[fid], c, e.stats) -
                    snode_[nid].root_gain); // TODO: important function, we can use snode_[nid].root_gain
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

      std::vector<unsigned> nid_split_index;
      nid_split_index.resize(num_node_,0);
      std::vector<float> nid_split_cond;
      nid_split_cond.resize(num_node_,0);

      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        fsplits.push_back(e.best.SplitIndex());
        nid_split_index.at(nid)=e.best.SplitIndex();
        nid_split_cond.at(nid)=e.best.split_value;
      }  
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      // std::cout<<"1"<<'\n';

      dmlc::DataIter<ColBatch> *iter_task = p_fmat->ColIterator(fsplits);
      while (iter_task->Next()) {  
        const ColBatch &batch = iter_task->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const auto ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = position_[ridx];
            std::cout<<nid<<'\t';      
            inst_nid_.at(ridx)=nid;
            const bst_float fvalue = col[j].fvalue;
            if (nid_split_index.at(nid) == fid) {
              if (fvalue < nid_split_cond.at(nid)) {
                inst_go_left_.at(ridx)=true;
              } else {
                inst_go_left_.at(ridx)=false;
              }
            }
          }
        }
      }
      std::cout<<"2"<<'\n';

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
            int task_value=int(fvalue);

            // const int nid = position_[ridx];
            // inst_nid_.at(ridx)=nid;
            inst_task_id_.at(ridx)=task_value;
          }
        }
    }

    inline void InitAuxiliary(const uint64_t num_row_, const int num_node_,const std::vector<int> &qexpand){
      /****************************** init auxiliary val ***********************************/
      // they can be init at the begining of each depth before the real task split

      // the nid of each inst
      // data_in
      inst_nid_.resize(num_row_,-1);
      // the task_id of each isnt
      inst_task_id_.resize(num_row_,-1);
      // whether the inst goes left (true) or right (false)
      inst_go_left_.resize(num_row_,true);

      // G and H in capital means the sum of the G and H over all the instances
      // store the G and H in the node, in order to calculate w* for the whole node 
      // auto biggest = std::max_element(std::begin(qexpand), std::end(qexpand));
      // std::cout<<*biggest<<"\t\n";
      G_node_.resize(num_node_);
      H_node_.resize(num_node_);
      for (int nid : qexpand) {
        G_node_.at(nid)=0;
        H_node_.at(nid)=0;
      }
      // store the G and H for each task in each node's left and right child, the 
      G_task_lnode_.resize(num_node_);
      G_task_rnode_.resize(num_node_);
      H_task_lnode_.resize(num_node_);
      H_task_rnode_.resize(num_node_);
      for (int nid : qexpand) {
        G_task_lnode_.at(nid).resize(task_num_for_init_vec,-1);
        G_task_rnode_.at(nid).resize(task_num_for_init_vec,-1);
        H_task_lnode_.at(nid).resize(task_num_for_init_vec,-1);
        H_task_rnode_.at(nid).resize(task_num_for_init_vec,-1);
        for (int task_id : tasks_list_){
          G_task_lnode_.at(nid).at(task_id)=0;
          G_task_rnode_.at(nid).at(task_id)=0;
          H_task_lnode_.at(nid).at(task_id)=0;
          H_task_rnode_.at(nid).at(task_id)=0;
        }  
      }

    }
    
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
      int node_isn_num[100];

      /****************************** init auxiliary val ***********************************/
      auto num_row=p_fmat->Info().num_row_;
      auto biggest = std::max_element(std::begin(qexpand), std::end(qexpand));
      int num_node = (*biggest+1);
      InitAuxiliary(num_row,num_node,qexpand_);
      // std::cout<<"init aux"<<'\n';

      // /******************  assign nid,  task_id,  go_left  for each inst *******************/
      AssignTaskAndNid(p_tree,p_fmat,feat_set,qexpand_,num_node);
      // for (int ind : inst_nid_){std::cout<<ind<<"\t";}
      // for (int ind : inst_task_id_){std::cout<<ind<<"\t";}
      // for (int ind : inst_go_left_){std::cout<<ind<<"\t";}
      // std::cout<<"assign"<<'\n';
      


      /************* starts of calculate G, H through the whole data*******/
      // step 0, resize the G,H,for node and task here following this :
      /*
      snode_.resize(tree.param.num_nodes, NodeEntry(param_)); // FIXME: learn this, 
      constraints_.resize(tree.param.num_nodes);
      */



      



      /************* end    of calculate G, H through the whole data*******/
      






      // for (int nid : qexpand) {

        



      //   //  NodeEntry &e = snode_[nid];
      //   NodeEntry &e = snode_[nid];
      //   auto best_feature_split_fidx = e.best.SplitIndex();
      //   auto best_feature_split_value = e.best.split_value;



      //   std::cout<<nid<<" th node here \n\n"<<'\n';
      //   dmlc::DataIter<ColBatch>* iter = p_fmat->ColIterator(feat_set); // how to get the data just at this node? 

      //   int ins_in_bacth_list[100]={0};
      //   int bacth_iter_cnt=0;
      //   int ins_num_in_batch=0;
      //   int task_cnt_list[30]={0};
      //   int task_G_list[30]={0};
      //   int task_H_list[30]={0};
        

      //   // define the G_L_task,H_L_task,G_R_task,h_R_task, G_task, H_task for further computing.
      //   float G_L_task=0,H_L_task=0,G_R_task=0,h_R_task=0,G_task=0,H_task=0;           
        
      //   while (iter->Next()) {
      //     auto batch = iter->Value();
      //     const auto nsize = static_cast<bst_omp_uint>(batch.size);
      //     bst_omp_uint task_i;
      //     for (bst_omp_uint i = 0; i < nsize; ++i) {
      //     const bst_uint fid = batch.col_index[i];  // should check the feature idx, i is not the idx.
      //     if (fid==0){
      //       task_i = i; // find the task idx task_i
      //       break;
      //       }
      //     }  
          
      //     ins_num_in_batch=0;
      //     // get the task data.
      //     const ColBatch::Inst task_c = batch[task_i];

      //     // check the data
      //     for (const ColBatch::Entry *it = task_c.data; it != task_c.data+task_c.length; it += 1) {
            
      //       const bst_uint ridx = it->index;


      //       const int ins_nid=position_[ridx];
      //       if (nid==ins_nid){ // make sure that the ins belongs to the corrent processing node with nid. FIXME, could be parallel LATER
      //         // const int nid = position_[ridx];
      //         const bst_float fvalue = it->fvalue;
      //         int task_value=int(fvalue);
      //         task_cnt_list[task_value]++;

      //         ins_num_in_batch++;
              
      //         // get the gradient here
      //         const GradientPair& b = gpair[ridx];
      //         G_task+=b.GetGrad();
      //         task_G_list[task_value]+=b.GetGrad();
              
      //         H_task+=b.GetHess();
      //         task_H_list[task_value]+=b.GetHess();

      //       }
      //     }
      //   bacth_iter_cnt++;
      //   }
      //   if (bacth_iter_cnt<100){
      //     ins_in_bacth_list[bacth_iter_cnt]=ins_num_in_batch;
      //     std::cout<<ins_num_in_batch<<'\n';
          
      //   }
        
      // }


      //=====================  end   of task split =======================

        for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > kRtEps) {
          p_tree->AddChilds(nid);
          (*p_tree)[nid].SetSplit(e.best.SplitIndex(), e.best.split_value, e.best.DefaultLeft());
          // mark right child as 0, to indicate fresh leaf
          (*p_tree)[(*p_tree)[nid].LeftChild()].SetLeaf(0.0f, 0);
          (*p_tree)[(*p_tree)[nid].RightChild()].SetLeaf(0.0f, 0);
        } else {
          (*p_tree)[nid].SetLeaf(e.weight * param_.learning_rate);
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
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       DMatrix *p_fmat,
                                       const RegTree &tree) { 
      // step 1, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        if (!tree[nid].IsLeaf()) {
          fsplits.push_back(tree[nid].SplitIndex());
        }
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator(fsplits);
      while (iter->Next()) {  //TODO: learn the code here!
        const ColBatch &batch = iter->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          ColBatch::Inst col = batch[i];
          const bst_uint fid = batch.col_index[i];
          const auto ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = this->DecodePosition(ridx);
            const bst_float fvalue = col[j].fvalue;
            // go back to parent, correct those who are not default
            if (!tree[nid].IsLeaf() && tree[nid].SplitIndex() == fid) {
              if (fvalue < tree[nid].SplitCond()) {
                this->SetEncodePosition(ridx, tree[nid].LeftChild());
              } else {
                this->SetEncodePosition(ridx, tree[nid].RightChild());
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
    std::vector< std::vector<ThreadEntry> > stemp_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode_;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
    // constraint value
    std::vector<TConstraint> constraints_;


    /****************************** auxiliary val ***********************************/
    // they can be init at the begining of each depth before the real task split

    // 
    const std::vector<int> tasks_list_{1, 2, 4, 5, 6, 10, 11, 12, 13, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    const int task_num_for_init_vec=30;

    // the nid of each inst
    std::vector<int> inst_nid_;

    // the task_id of each isnt
    std::vector<int> inst_task_id_;

    // whether the inst goes left (true) or right (false)
    std::vector<bool> inst_go_left_;


    //TODO: how to init the vec in vec ?
    // note that the index starts from 0
    // G and H in capital means the sum of the G and H over all the instances
    // store the G and H in the node, in order to calculate w* for the whole node 
    std::vector<float> G_node_;
    std::vector<float> H_node_;

    // store the G and H for each task in each node's left and right child, the 
    std::vector<std::vector<float> > G_task_lnode_;
    std::vector<std::vector<float> > G_task_rnode_;
    std::vector<std::vector<float> > H_task_lnode_;
    std::vector<std::vector<float> > H_task_rnode_; //TODO: init 
    
    
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
