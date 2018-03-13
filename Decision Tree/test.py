from subprocess import check_call

check_call(['dot','-Tpng','decision_tree.dot','-o','decision_tree.png'])