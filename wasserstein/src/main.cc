//
// Labels must be nonnegative integers 
//


#include <iostream>
#include <fstream>
#include <lemon/smart_graph.h>
#include <lemon/network_simplex.h>
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;
using namespace lemon;

// User must define feature and distance function
using MyFeature = vector<pair<int, double>>; // = sparse vector
struct MyDatum {
  int label;
  MyFeature feature;
};
double distance(MyFeature x, MyFeature y) {
  auto square = [&](double t) { return t * t; };
  double score = 0;
  for (int i = 0, j = 0; i != x.size() || j != y.size(); ) {
    if (i == x.size()) {
      score += square(y[j].second);
      ++j;
    } else if (j == y.size()) {
      score += square(x[i].second);
      ++i;
    } else if (x[i].first < y[j].first) {
      score += square(x[i].second);
      ++i;
    } else if (x[i].first > y[j].first) {
      score += square(y[j].second);
      ++j;
    } else {
      score += square(x[i].second - y[j].second);
      ++i;
      ++j;
    } 
  }
  return score;
}
vector<MyDatum> readFile(string filename) {
  vector<MyDatum> data;
  ifstream ifs(filename);
  
  for (string line; getline(ifs, line); ) {
    if (line[0] == '#') continue;
    MyDatum datum;
    for (int i = 0; i < line.size(); ++i) 
      if (line[i] == ':') line[i] = ' ';
    stringstream ss(line);
    ss >> datum.label;
    int index;
    double value;
    while (ss >> index >> value) {
      datum.feature.push_back(make_pair(index, value));
    }
    sort(datum.feature.begin(), datum.feature.end());
    data.push_back(datum);
  }
  return data;
}

// sizes[label] = number of points having the label
template <class Datum>
double wasserstein(vector<Datum> data1, vector<Datum> data2) {
  int K = 0, N1 = data1.size(), N2 = data2.size();

  SmartDigraph g;
  SmartDigraph::ArcMap<int> capacity(g);
  SmartDigraph::ArcMap<double> cost(g);

  // supersource, superterminal
  SmartDigraph::Node s = g.addNode();
  SmartDigraph::Node t = g.addNode();

  // left vertices
  vector<SmartDigraph::Node> left;
  vector<SmartDigraph::Arc> incomming;
  for (int i = 0; i < data1.size(); ++i) {
    left.push_back(g.addNode());
    SmartDigraph::Arc a = g.addArc(s, left[i]);
    capacity[a] = N2;
    cost[a] = 0;
    incomming.push_back(a);
  }
  // right vertices
  vector<SmartDigraph::Node> right;
  for (int i = 0; i < data2.size(); ++i) {
    right.push_back(g.addNode());
    SmartDigraph::Arc a = g.addArc(right[i], t);
    capacity[a] = N1;
    cost[a] = 0;
  }

  for (int i = 0; i < left.size(); ++i) {
    for (int j = 0; j < right.size(); ++j) {
      SmartDigraph::Arc a = g.addArc(left[i], right[j]);
      capacity[a] = N1 * N2;
      cost[a] = distance(data1[i].feature, data2[j].feature);
    }
  }

  NetworkSimplex<SmartDigraph, int, double> ns(g);
  ns.upperMap(capacity);
  ns.costMap(cost);
  ns.stSupply(s, t, N1 * N2);

  bool res = ns.run();
  if (!res) cerr << "infeasible" << endl;
  SmartDigraph::ArcMap<int> flow(g);
  ns.flowMap(flow);
  return ns.totalCost() / (N1 * N2);
}


int main(int argc, char *argv[]) {
  vector<MyDatum> data1 = readFile(argv[1]);
  vector<MyDatum> data2 = readFile(argv[2]);
  double totalCost = wasserstein(data1, data2);
  cout << totalCost << endl;
}
