#ifndef MODEL_H
#define MODEL_H

#include <list>
#include <map>

class Model {
  
  std::list<std::pair<std::string, util::timestamp> > complete;
  std::map<std::string, util::timestamp> pending;

public:
  void begin(const std::string& s) {
    pending[s] = util::timestamp();
  }

  void end (const std::string& s) {
    complete.push_back(std::make_pair (s, util::timestamp()-pending[s]));
  }

  void dump (std::ostream& out = std::cout) {
    for (auto pa : complete) 
      out<<pa.first<<" : "<<pa.second<<std::endl;
  }
};

#endif
