pwd=$PWD

array=(element0 element1 element2)

for(( i=0;i<${#array[@]};i++)) 
do
echo ${array[i]}
done


for element in ${array[@]}
do
echo $pwd/$element
done

for i in "${!array[@]}"
do
echo ${array[i]}
done