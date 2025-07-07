#include "AST/function.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>

std::string removeNonUnicode(const std::string& input) {
    std::string result;
    std::copy_if(input.begin(), input.end(), std::back_inserter(result), [](char c) {
        return static_cast<unsigned char>(c) < 128;
    });
    return result;
}

// TODO
FunctionNode::FunctionNode(const uint32_t line, const uint32_t col,
                            const char* p_name, const char* p_rtn_type, std::vector<AstNode*> *p_decl_list, AstNode* p_compound)
    : AstNode{line, col}, name(p_name), rtn_type(p_rtn_type), decl_list(p_decl_list), compound(p_compound) {}

// TODO: You may use code snippets in AstDumper.cpp
void FunctionNode::print() {}

void FunctionNode::accept(AstNodeVisitor &p_visitor){
    p_visitor.visit(*this);
}

void FunctionNode::visitChildNodes(AstNodeVisitor &p_visitor) {
    if(decl_list != nullptr){
        for(auto &decl : *decl_list){
            decl->accept(p_visitor);
        }
    }
    if(compound != nullptr){
        compound->accept(p_visitor);
    }
}

const char* FunctionNode::getNameCString() const{
    return name.c_str();
}

const char* FunctionNode::getRtnTypeCString() const{
    return rtn_type.c_str();
}

std::string FunctionNode::getPrototypeString(){
    DeclNode* d;
    VariableNode *v;
    std::stringstream rtnStream;
    rtnStream << "(";
    //std::cout << "Intermediate String: " << rtnStream.str() << std::endl;

    if(decl_list != nullptr){
        for(uint32_t i = 0;i<decl_list->size();i++){
            d = dynamic_cast<DeclNode*>(decl_list->at(i));
            std::vector<AstNode*> *var_list = d->getVarList();
            if(i != 0){
                rtnStream << ", ";
            }
            //std::cout << "Intermediate String: " << rtnStream.str() << std::endl;

            if(var_list != nullptr){
                uint32_t lth = var_list->size();
                for(uint32_t j = 0;j<lth;j++){
                    v = dynamic_cast<VariableNode*>(var_list->at(j));
                    
                    rtnStream << v->getType().c_str();
                    if(j != lth - 1){
                        rtnStream << ", ";
                    }
                }
            }
            //std::cout << "Intermediate String: " << rtnStream.str() << std::endl;

        }
    }
    //std::cout << "Intermediate String: " << rtnStream.str() << std::endl;

    rtnStream << ")";
    //std::cout.flush();
    //std::cout << "Intermediate String: " << rtnStream.str() << std::endl;

    //printf("%s", rtnStream.str());
    return rtnStream.str();
}
