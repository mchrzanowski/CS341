#include <iostream>
#include <armadillo>
#include <unordered_set>
#include <unordered_map>
#include "strtk.hpp"
#include <vector>
#include <boost/algorithm/string.hpp>


using namespace std;
using namespace arma;

class RBM {

    constexpr static float epsilonw = 0.001;
    constexpr static float epsilonvb = 0.008;
    constexpr static float epsilonhb = 0.0006;
    constexpr static float weight_cost = 0.0001;
    constexpr static float momentum = 0.8;
    constexpr static float final_momentum = 0.9;

    string training_file;

    unordered_map<long, int> item_translation;
    unordered_map<long, int> user_translation;
    unordered_set<int> labels;

    bool verbose;

    int batch_size;
    int hidden_units;

    unordered_map<int, vector<tuple<int, int>>> users_to_items;

    mat item_rating_count;

    mat visible_biases;
    vec hidden_biases;

    cube weights;

    int ITEMS;
    int FEATURES;
    int SOFTMAX;


public:

    RBM(): training_file("./training"), hidden_units(18), verbose(false), batch_size(100) {

        users_to_items = RBM::parse_input_file(training_file, false);

        ITEMS = item_translation.size();
        FEATURES = hidden_units;
        SOFTMAX = labels.size();

        item_rating_count = zeros<mat>(ITEMS, SOFTMAX);

        hidden_biases = zeros<vec>(FEATURES);

        weights = 0.02 * randn<cube>(ITEMS, SOFTMAX, FEATURES) - 0.01;

        populate_with_input_data();

    }

    unordered_map<int, vector<tuple<int, int>>> parse_input_file(string filename, bool exclude_unseen){

        unordered_map<int, vector<tuple<int, int>>> user_to_items;

        ifstream ifs;
        ifs.open(filename, ifstream::in);

        while (ifs.good()){
            string line;
            getline(ifs, line);

            vector<string> values;
            boost::split(values, line, boost::is_any_of("\t"));

            if (values.size() != 3){
                break;
            }

            long user = stol(values[0]);
            long item = stol(values[1]);
            int rating = stoi(values[2]);

            if (user_translation.find(user) == user_translation.end() && ! exclude_unseen){
                user_translation[user] = user_translation.size();
            }

            if (item_translation.find(item) == item_translation.end() && ! exclude_unseen){
                item_translation[item] = item_translation.size();
            }

            if (labels.find(rating) == labels.end() && ! exclude_unseen){
                labels.insert(rating);
            }

            if (user_translation.find(user) != user_translation.end() && item_translation.find(item) != item_translation.end() && labels.find(rating) != labels.end()){
                int translated_user = user_translation[user];
                if (user_to_items.find(translated_user) == user_to_items.end()){
                    user_to_items[translated_user] = vector<tuple<int, int>>();
                }
                int translated_item = item_translation[item];
                user_to_items[translated_user].push_back(make_tuple(translated_item, rating));
            }
        }
        
        return user_to_items;
    }

    void populate_with_input_data() {

        for (auto kv: users_to_items){
            for (auto v : kv.second){
                this->item_rating_count[get<0>(v), get<1>(v)] += 1;
            }
        }

        mat non_zero_item_rating_count = mat(this->item_rating_count);
        auto zero_elements = find(non_zero_item_rating_count == 0);
        non_zero_item_rating_count(zero_elements) += 1;

        colvec normalization = sum(this->item_rating_count, 1);
        auto zero_norms = find(normalization == 0);
        normalization(zero_norms) += 1;

        mat m_normalization = repmat(normalization, 1, this->SOFTMAX);

        this->visible_biases = non_zero_item_rating_count / m_normalization;
        this->visible_biases = log(this->visible_biases);

        auto activations = find(item_rating_count == 0);
        this->visible_biases(activations) *= 0;

    }
};



int main(int argc, char** argv){
    /*mat A;
    A.randu(5, 5);

    colvec B;
    B.ones(5);

    mat D = diagmat(B);

    cout << A << endl;
    cout << D * A << endl;

    unordered_set<int> C;
    C.insert(3);

    cout << C.size() << endl;
    */

    RBM r;
    
    return 0;
}

