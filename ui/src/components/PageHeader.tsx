interface Props {
  title: string;
  subtitle: string;
}

export function PageHeader({ title, subtitle }: Props) {
  return (
    <div>
      <h1 className="title-gradient page-title">{title}</h1>
      <p className="page-subtitle mt-2.5">{subtitle}</p>
    </div>
  );
}
