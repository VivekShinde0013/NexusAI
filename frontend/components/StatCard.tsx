import { LucideIcon } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  description: string;
  icon: LucideIcon;
  trend?: string;
  color?: "blue" | "indigo" | "emerald" | "purple";
}

export function StatCard({
  title,
  value,
  description,
  icon: Icon,
  trend,
  color = "blue",
}: StatCardProps) {
  const colorStyles = {
    blue: "from-blue-500/10 to-blue-600/5 text-blue-400 border-blue-500/20",
    indigo: "from-indigo-500/10 to-indigo-600/5 text-indigo-400 border-indigo-500/20",
    emerald: "from-emerald-500/10 to-emerald-600/5 text-emerald-400 border-emerald-500/20",
    purple: "from-purple-500/10 to-purple-600/5 text-purple-400 border-purple-500/20",
  };

  return (
    <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700/80 transition-all duration-200 shadow-lg shadow-black/20 relative overflow-hidden group">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400 tracking-wide uppercase">
          {title}
        </span>
        <div className={`p-2.5 rounded-xl bg-gradient-to-br border ${colorStyles[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>

      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-2xl sm:text-3xl font-extrabold text-slate-100 tracking-tight">
          {value}
        </span>
        {trend && (
          <span className="text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-500/20">
            {trend}
          </span>
        )}
      </div>

      <p className="mt-1.5 text-xs text-slate-400 leading-relaxed">
        {description}
      </p>

      {/* Decorative gradient glow on hover */}
      <div className="absolute -bottom-10 -right-10 w-24 h-24 bg-blue-500/5 rounded-full blur-2xl group-hover:bg-blue-500/10 transition-colors pointer-events-none" />
    </div>
  );
}
