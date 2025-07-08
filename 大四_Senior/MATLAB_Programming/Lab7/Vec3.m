classdef Vec3
    properties
        x = 0;
        y = 0;
        z = 0;

    end
    methods
        function obj = Vec3(a, b, c)
            if nargin == 0
                obj.x = 0;
                obj.y = 0;
                obj.z = 0;
            elseif nargin == 3
                validateattributes(a, {'numeric'}, {'scalar'});
                validateattributes(b, {'numeric'}, {'scalar'});
                validateattributes(c, {'numeric'}, {'scalar'});
                obj.x = a;
                obj.y = b;
                obj.z = c;
            else 
                error('Expected input to be scalar');
            end

        end

        function f = plus(f1, f2)
            if isscalar(f1) && isscalar(f2)
                f = reshape(Vec3([f1.x]+[f2.x], [f1.y]+[f2.y], [f1.z]+[f2.z]), size(f1));
            end
        end
        function f = minus(f1, f2)
            if isscalar(f1) && isscalar(f2)
                f = reshape(Vec3([f1.x]-[f2.x], [f1.y]-[f2.y], [f1.z]-[f2.z]), size(f1));
            end
        end
        function f = inner_prod(f1, f2)
            if isscalar(f1) && isscalar(f2)
                f = [f1.x]*[f2.x]+[f1.y]*[f2.y]+[f1.z]*[f2.z];
            end
        end

        function v = norm(p)
            v = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        end

        function disp(ob)
            fprintf('(%d,%d,%d)\n', ob.x, ob.y, ob.z);
        end


    end

end