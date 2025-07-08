#ifndef AST_P_TYPE_H
#define AST_P_TYPE_H

#include <memory>
#include <string>
#include <vector>

class PType;

using PTypeSharedPtr = std::shared_ptr<PType>;
class PType{
  public:
    enum class PrimitiveTypeEnum : uint8_t{
        kVoidType,
        kIntegerType,
        kRealType,
        kBoolType,
        kStringType,
        kErrorType
    };

  private:
    PrimitiveTypeEnum m_type;
    std::vector<uint64_t> m_dimensions;
    mutable std::string m_type_string;
    mutable bool m_type_string_is_valid = false;

  public:
    ~PType() = default;
    PType(const PrimitiveTypeEnum type) : m_type(type){}

    void setDimensions(std::vector<uint64_t> &p_dims){
        m_dimensions = std::move(p_dims);
    }

    PrimitiveTypeEnum getPrimitiveType() const{ 
      return m_type; 
    }
    const char *getPTypeCString() const;

    bool zeroDimension(){
      for(auto &d : m_dimensions){
        if(d < 1) {
          return true;
        }
      }
      return false;
    }

    int getDimensionsSize(){ 
      return m_dimensions.size(); 
    }
    std::vector<uint64_t> getDimensions(){ 
      return m_dimensions; 
    }

    std::string getType(){
      if(m_type == PrimitiveTypeEnum::kVoidType){
        return "void";
      } else if(m_type == PrimitiveTypeEnum::kIntegerType){
        return "integer";
      } else if(m_type == PrimitiveTypeEnum::kRealType){
        return "real";
      } else if(m_type == PrimitiveTypeEnum::kBoolType){
        return "boolean";
      } else if(m_type == PrimitiveTypeEnum::kStringType){
        return "string";
      } else{
        return "error";
      }
    }

};

#endif
